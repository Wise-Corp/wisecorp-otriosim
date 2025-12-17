/*
 * Find Winning Strategy - Parallel Minimax
 *
 * Explores the game tree and computes winning strategy using minimax,
 * all in one pass. No JSON intermediate file needed.
 *
 * Compile: gcc -O3 -march=native -o find_strategy find_strategy.c -lpthread
 * Usage:   ./find_strategy --depth 7 --parallel 32
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <time.h>

#define POS_COUNT 9
#define SIZE_COUNT 3
#define PLAYER_BLUE 0
#define PLAYER_RED 1

#define BIT(pos, size) (1U << ((pos) * 3 + (size)))

static const int WIN_LINES[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6}
};

static const char *POS_NAMES[9] = {"A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"};
static const char SIZE_CHARS[3] = {'S', 'M', 'L'};

typedef struct {
    uint32_t blue;
    uint32_t red;
    uint8_t blue_hand;
    uint8_t red_hand;
    uint8_t turn;
} State;

typedef struct {
    uint8_t pos;
    uint8_t size;
} Move;

/* Result of minimax */
typedef enum {
    RESULT_UNKNOWN = 0,  /* Depth limit reached */
    RESULT_BLUE_WINS = 1,
    RESULT_RED_WINS = -1,
} Result;

/* Strategy node */
typedef struct StrategyNode {
    char move[4];
    Result result;
    uint32_t num_children;
    struct StrategyNode *children;
} StrategyNode;

/* Global config */
static int g_max_depth = 0;
static int g_num_workers = 1;

/* Statistics */
typedef struct {
    uint64_t nodes;
    uint64_t blue_wins;
    uint64_t red_wins;
    uint64_t depth_limit;
} Stats;

static inline void state_init(State *s) {
    s->blue = 0;
    s->red = 0;
    s->blue_hand = 0xDB;
    s->red_hand = 0xDB;
    s->turn = PLAYER_BLUE;
}

static inline int hand_count(uint8_t hand, int size) {
    return (hand >> (size * 3)) & 0x7;
}

static inline uint8_t hand_dec(uint8_t hand, int size) {
    return hand - (1 << (size * 3));
}

static inline bool is_occupied(const State *s, int pos, int size) {
    uint32_t bit = BIT(pos, size);
    return (s->blue & bit) || (s->red & bit);
}

static inline bool check_same_size_win(uint32_t board, int size) {
    for (int line = 0; line < 8; line++) {
        int p0 = WIN_LINES[line][0], p1 = WIN_LINES[line][1], p2 = WIN_LINES[line][2];
        if ((board & BIT(p0, size)) && (board & BIT(p1, size)) && (board & BIT(p2, size)))
            return true;
    }
    return false;
}

static inline bool check_nested_win(uint32_t board) {
    for (int pos = 0; pos < 9; pos++) {
        if ((board & BIT(pos, 0)) && (board & BIT(pos, 1)) && (board & BIT(pos, 2)))
            return true;
    }
    return false;
}

static inline bool check_sequence_win(uint32_t board) {
    for (int line = 0; line < 8; line++) {
        int p0 = WIN_LINES[line][0], p1 = WIN_LINES[line][1], p2 = WIN_LINES[line][2];
        if ((board & BIT(p0, 0)) && (board & BIT(p1, 1)) && (board & BIT(p2, 2)))
            return true;
        if ((board & BIT(p0, 2)) && (board & BIT(p1, 1)) && (board & BIT(p2, 0)))
            return true;
    }
    return false;
}

static inline bool has_won(uint32_t board) {
    return check_same_size_win(board, 0) ||
           check_same_size_win(board, 1) ||
           check_same_size_win(board, 2) ||
           check_nested_win(board) ||
           check_sequence_win(board);
}

static int generate_moves(const State *s, Move *moves) {
    int count = 0;
    uint8_t hand = (s->turn == PLAYER_BLUE) ? s->blue_hand : s->red_hand;
    for (int size = 0; size < 3; size++) {
        if (hand_count(hand, size) == 0) continue;
        for (int pos = 0; pos < 9; pos++) {
            if (!is_occupied(s, pos, size)) {
                moves[count].pos = pos;
                moves[count].size = size;
                count++;
            }
        }
    }
    return count;
}

static inline void apply_move(State *s, Move m) {
    uint32_t bit = BIT(m.pos, m.size);
    if (s->turn == PLAYER_BLUE) {
        s->blue |= bit;
        s->blue_hand = hand_dec(s->blue_hand, m.size);
    } else {
        s->red |= bit;
        s->red_hand = hand_dec(s->red_hand, m.size);
    }
    s->turn = 1 - s->turn;
}

/*
 * Minimax with strategy extraction.
 * Returns the result and builds strategy tree for winning paths.
 *
 * strategy: if non-NULL and BLUE wins, fills in the winning strategy node
 */
static Result minimax(State *s, int depth, Stats *stats, StrategyNode *strategy) {
    stats->nodes++;

    /* Check for win by previous player */
    if (s->turn == PLAYER_RED && has_won(s->blue)) {
        stats->blue_wins++;
        if (strategy) strategy->result = RESULT_BLUE_WINS;
        return RESULT_BLUE_WINS;
    }
    if (s->turn == PLAYER_BLUE && has_won(s->red)) {
        stats->red_wins++;
        if (strategy) strategy->result = RESULT_RED_WINS;
        return RESULT_RED_WINS;
    }

    Move moves[27];
    int num_moves = generate_moves(s, moves);

    /* No moves = draw (treat as unknown) */
    if (num_moves == 0) {
        if (strategy) strategy->result = RESULT_UNKNOWN;
        return RESULT_UNKNOWN;
    }

    /* Depth limit */
    if (g_max_depth > 0 && depth >= g_max_depth) {
        stats->depth_limit++;
        if (strategy) strategy->result = RESULT_UNKNOWN;
        return RESULT_UNKNOWN;
    }

    bool is_blue_turn = (s->turn == PLAYER_BLUE);

    if (is_blue_turn) {
        /* BLUE's turn: find ANY winning move */
        Result best = RESULT_RED_WINS;
        int best_idx = -1;
        StrategyNode *best_child = NULL;

        for (int i = 0; i < num_moves; i++) {
            State child = *s;
            apply_move(&child, moves[i]);

            StrategyNode child_strat = {0};
            Result r = minimax(&child, depth + 1, stats, strategy ? &child_strat : NULL);

            if (r > best) {
                best = r;
                best_idx = i;
                if (strategy && best_child) {
                    /* Free previous best */
                    free(best_child->children);
                }
                if (strategy) {
                    best_child = malloc(sizeof(StrategyNode));
                    *best_child = child_strat;
                    snprintf(best_child->move, 4, "%c%s", SIZE_CHARS[moves[i].size], POS_NAMES[moves[i].pos]);
                }
            } else if (strategy && r != best) {
                /* Not best, free it */
                free(child_strat.children);
            }

            if (best == RESULT_BLUE_WINS) break;
        }

        if (strategy) {
            strategy->result = best;
            if (best == RESULT_BLUE_WINS && best_child) {
                strategy->num_children = 1;
                strategy->children = best_child;
            } else {
                if (best_child) free(best_child);
            }
        }
        return best;

    } else {
        /* RED's turn: BLUE wins only if ALL moves lead to BLUE win */
        Result worst = RESULT_BLUE_WINS;
        StrategyNode *children = NULL;
        int num_winning = 0;
        int children_cap = 0;

        for (int i = 0; i < num_moves; i++) {
            State child = *s;
            apply_move(&child, moves[i]);

            StrategyNode child_strat = {0};
            Result r = minimax(&child, depth + 1, stats, strategy ? &child_strat : NULL);

            if (r < worst) {
                worst = r;
                /* RED found escape, can stop if not building strategy */
                if (!strategy && worst == RESULT_RED_WINS) break;
            }

            if (strategy && r == RESULT_BLUE_WINS) {
                /* Add to children */
                if (num_winning >= children_cap) {
                    children_cap = children_cap ? children_cap * 2 : 8;
                    children = realloc(children, children_cap * sizeof(StrategyNode));
                }
                children[num_winning] = child_strat;
                snprintf(children[num_winning].move, 4, "%c%s", SIZE_CHARS[moves[i].size], POS_NAMES[moves[i].pos]);
                num_winning++;
            } else if (strategy) {
                free(child_strat.children);
            }
        }

        if (strategy) {
            strategy->result = worst;
            if (worst == RESULT_BLUE_WINS) {
                strategy->num_children = num_winning;
                strategy->children = children;
            } else {
                /* RED can escape, free children */
                for (int i = 0; i < num_winning; i++) {
                    free(children[i].children);
                }
                free(children);
            }
        }
        return worst;
    }
}

/* Output strategy as JSON */
static void output_json(FILE *f, StrategyNode *node, bool first) {
    if (!first) fprintf(f, ",");

    if (node->num_children == 0) {
        fprintf(f, "{\"m\":\"%s\",\"r\":\"B\"}", node->move);
    } else {
        if (node->move[0]) {
            fprintf(f, "{\"m\":\"%s\",\"c\":[", node->move);
        } else {
            fprintf(f, "{\"c\":[");
        }
        for (uint32_t i = 0; i < node->num_children; i++) {
            output_json(f, &node->children[i], i == 0);
        }
        fprintf(f, "]}");
    }
}

/* Output paths as text */
static void output_paths(FILE *f, StrategyNode *node, char *path, int path_len) {
    int new_len = path_len;
    if (node->move[0]) {
        if (path_len > 0) {
            memcpy(path + path_len, " -> ", 4);
            path_len += 4;
        }
        strcpy(path + path_len, node->move);
        new_len = path_len + strlen(node->move);
    }

    if (node->num_children == 0) {
        path[new_len] = 0;
        fprintf(f, "%s\n", path);
    } else {
        for (uint32_t i = 0; i < node->num_children; i++) {
            output_paths(f, &node->children[i], path, new_len);
        }
    }
}

static void free_strategy(StrategyNode *node) {
    for (uint32_t i = 0; i < node->num_children; i++) {
        free_strategy(&node->children[i]);
    }
    free(node->children);
}

/* Worker data for parallel processing */
typedef struct {
    Move move;
    State state;
    Stats stats;
    Result result;
    StrategyNode strategy;
} WorkerData;

static void *worker_thread(void *arg) {
    WorkerData *w = (WorkerData *)arg;
    memset(&w->stats, 0, sizeof(Stats));
    memset(&w->strategy, 0, sizeof(StrategyNode));

    w->result = minimax(&w->state, 1, &w->stats, &w->strategy);
    snprintf(w->strategy.move, 4, "%c%s", SIZE_CHARS[w->move.size], POS_NAMES[w->move.pos]);

    return NULL;
}

int main(int argc, char **argv) {
    const char *json_file = NULL;
    const char *paths_file = NULL;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--depth") == 0) && i + 1 < argc) {
            g_max_depth = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--parallel") == 0) && i + 1 < argc) {
            g_num_workers = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            json_file = argv[++i];
        } else if (strcmp(argv[i], "--paths") == 0 && i + 1 < argc) {
            paths_file = argv[++i];
        }
    }

    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "OTRIO WINNING STRATEGY FINDER (parallel minimax)\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "Workers: %d\n", g_num_workers);
    if (g_max_depth > 0) fprintf(stderr, "Max depth: %d\n", g_max_depth);
    if (json_file) fprintf(stderr, "Output JSON: %s\n", json_file);
    if (paths_file) fprintf(stderr, "Output paths: %s\n", paths_file);
    fprintf(stderr, "\n");

    State initial;
    state_init(&initial);
    Move opening_moves[27];
    int num_opening = generate_moves(&initial, opening_moves);

    fprintf(stderr, "Parallelizing %d opening moves across %d workers...\n\n", num_opening, g_num_workers);

    clock_t start = clock();

    /* Prepare workers */
    WorkerData *workers = malloc(num_opening * sizeof(WorkerData));
    for (int i = 0; i < num_opening; i++) {
        workers[i].move = opening_moves[i];
        workers[i].state = initial;
        apply_move(&workers[i].state, opening_moves[i]);
    }

    /* Run in batches */
    pthread_t *threads = malloc(g_num_workers * sizeof(pthread_t));
    int completed = 0;

    while (completed < num_opening) {
        int batch = (num_opening - completed < g_num_workers) ?
                    (num_opening - completed) : g_num_workers;

        for (int i = 0; i < batch; i++) {
            pthread_create(&threads[i], NULL, worker_thread, &workers[completed + i]);
        }

        for (int i = 0; i < batch; i++) {
            pthread_join(threads[i], NULL);
            WorkerData *w = &workers[completed + i];
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            const char *res_str = (w->result == RESULT_BLUE_WINS) ? "BLUE" :
                                  (w->result == RESULT_RED_WINS) ? "RED" : "???";
            fprintf(stderr, "  %c%s: %s  nodes=%lu B=%lu R=%lu L=%lu [%.1fs]\n",
                    SIZE_CHARS[w->move.size], POS_NAMES[w->move.pos], res_str,
                    w->stats.nodes, w->stats.blue_wins, w->stats.red_wins,
                    w->stats.depth_limit, elapsed);
        }
        completed += batch;
    }

    free(threads);

    /* Aggregate results */
    Stats total = {0};
    int blue_winning_moves = 0;

    for (int i = 0; i < num_opening; i++) {
        total.nodes += workers[i].stats.nodes;
        total.blue_wins += workers[i].stats.blue_wins;
        total.red_wins += workers[i].stats.red_wins;
        total.depth_limit += workers[i].stats.depth_limit;
        if (workers[i].result == RESULT_BLUE_WINS) blue_winning_moves++;
    }

    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "Nodes explored: %lu\n", total.nodes);
    fprintf(stderr, "BLUE wins: %lu\n", total.blue_wins);
    fprintf(stderr, "RED wins: %lu\n", total.red_wins);
    fprintf(stderr, "Depth limit: %lu\n", total.depth_limit);
    fprintf(stderr, "Time: %.1fs\n", elapsed);
    fprintf(stderr, "\n");

    if (blue_winning_moves > 0) {
        fprintf(stderr, "RESULT: BLUE HAS %d WINNING OPENING MOVE(S)!\n", blue_winning_moves);

        /* Build root strategy */
        StrategyNode root = {0};
        root.result = RESULT_BLUE_WINS;
        root.children = malloc(blue_winning_moves * sizeof(StrategyNode));
        root.num_children = 0;

        for (int i = 0; i < num_opening; i++) {
            if (workers[i].result == RESULT_BLUE_WINS) {
                root.children[root.num_children++] = workers[i].strategy;
            } else {
                free_strategy(&workers[i].strategy);
            }
        }

        /* Count nodes in strategy */
        uint64_t strategy_nodes = 0;
        void count_nodes(StrategyNode *n) {
            strategy_nodes++;
            for (uint32_t i = 0; i < n->num_children; i++) {
                count_nodes(&n->children[i]);
            }
        }
        count_nodes(&root);
        fprintf(stderr, "Winning strategy: %lu nodes\n", strategy_nodes);

        if (json_file) {
            fprintf(stderr, "\nWriting strategy to %s...\n", json_file);
            FILE *f = fopen(json_file, "w");
            if (f) {
                fprintf(f, "{\"tree\":");
                output_json(f, &root, true);
                fprintf(f, "}\n");
                fclose(f);
                fprintf(stderr, "Done!\n");
            }
        }

        if (paths_file) {
            fprintf(stderr, "\nWriting paths to %s...\n", paths_file);
            FILE *f = fopen(paths_file, "w");
            if (f) {
                char path[1024];
                output_paths(f, &root, path, 0);
                fclose(f);
                fprintf(stderr, "Done!\n");
            }
        }

        free_strategy(&root);
    } else {
        fprintf(stderr, "RESULT: No guaranteed win for BLUE at depth %d\n", g_max_depth);
        fprintf(stderr, "  (Need deeper exploration)\n");

        for (int i = 0; i < num_opening; i++) {
            free_strategy(&workers[i].strategy);
        }
    }

    fprintf(stderr, "============================================================\n");

    free(workers);
    return 0;
}
