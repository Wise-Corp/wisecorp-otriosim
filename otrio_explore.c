/*
 * Otrio Game Tree Explorer - Optimized C Implementation
 *
 * Compile: gcc -O3 -march=native -o otrio_explore otrio_explore.c -lpthread
 * Usage:   ./otrio_explore --depth 7 --parallel 32 -o depth7.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <pthread.h>
#include <time.h>

/* Board representation:
 * - 9 positions (A1-C3), 3 sizes (S/M/L), 2 players
 * - Each player's board: 27 bits (9 positions * 3 sizes)
 * - State: 54 bits total + 1 bit for turn
 */

#define POS_COUNT 9
#define SIZE_COUNT 3
#define PLAYER_BLUE 0
#define PLAYER_RED 1

/* Bit positions: pos * 3 + size */
#define BIT(pos, size) (1U << ((pos) * 3 + (size)))

/* Win lines (indices into positions 0-8) */
static const int WIN_LINES[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},  /* rows */
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},  /* cols */
    {0, 4, 8}, {2, 4, 6}              /* diags */
};

/* Position names for output */
static const char *POS_NAMES[9] = {"A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"};
static const char SIZE_CHARS[3] = {'S', 'M', 'L'};

/* Game state - compact 64-bit representation */
typedef struct {
    uint32_t blue;      /* Blue's pieces on board (27 bits) */
    uint32_t red;       /* Red's pieces on board (27 bits) */
    uint8_t blue_hand;  /* Blue's remaining pieces: bits 0-2=S, 3-5=M, 6-8=L (counts) */
    uint8_t red_hand;   /* Red's remaining pieces */
    uint8_t turn;       /* 0=Blue, 1=Red */
} State;

/* Statistics */
typedef struct {
    uint64_t nodes;
    uint64_t blue_wins;
    uint64_t red_wins;
    uint64_t draws;
    uint64_t depth_limit;
} Stats;

/* Move representation */
typedef struct {
    uint8_t pos;   /* 0-8 */
    uint8_t size;  /* 0-2 */
} Move;

/* Global config */
static int g_max_depth = 0;
static int g_num_workers = 1;
static FILE *g_output = NULL;
static pthread_mutex_t g_output_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Initialize starting state */
static inline void state_init(State *s) {
    s->blue = 0;
    s->red = 0;
    s->blue_hand = 0x49;  /* 3 of each: 001 001 001 in groups of 3 bits = 0b001001001 */
    s->red_hand = 0x49;
    s->turn = PLAYER_BLUE;
}

/* Get count of size in hand */
static inline int hand_count(uint8_t hand, int size) {
    return (hand >> (size * 3)) & 0x7;
}

/* Decrement count in hand */
static inline uint8_t hand_dec(uint8_t hand, int size) {
    return hand - (1 << (size * 3));
}

/* Check if position+size is occupied (by either player) */
static inline bool is_occupied(const State *s, int pos, int size) {
    uint32_t bit = BIT(pos, size);
    return (s->blue & bit) || (s->red & bit);
}

/* Check for win - same size in a line */
static inline bool check_same_size_win(uint32_t board, int size) {
    for (int line = 0; line < 8; line++) {
        int p0 = WIN_LINES[line][0], p1 = WIN_LINES[line][1], p2 = WIN_LINES[line][2];
        if ((board & BIT(p0, size)) && (board & BIT(p1, size)) && (board & BIT(p2, size))) {
            return true;
        }
    }
    return false;
}

/* Check for win - nested (all 3 sizes in one position) */
static inline bool check_nested_win(uint32_t board) {
    for (int pos = 0; pos < 9; pos++) {
        if ((board & BIT(pos, 0)) && (board & BIT(pos, 1)) && (board & BIT(pos, 2))) {
            return true;
        }
    }
    return false;
}

/* Check for win - ascending/descending sizes in a line */
static inline bool check_sequence_win(uint32_t board) {
    for (int line = 0; line < 8; line++) {
        int p0 = WIN_LINES[line][0], p1 = WIN_LINES[line][1], p2 = WIN_LINES[line][2];
        /* Ascending: S at p0, M at p1, L at p2 */
        if ((board & BIT(p0, 0)) && (board & BIT(p1, 1)) && (board & BIT(p2, 2))) {
            return true;
        }
        /* Descending: L at p0, M at p1, S at p2 */
        if ((board & BIT(p0, 2)) && (board & BIT(p1, 1)) && (board & BIT(p2, 0))) {
            return true;
        }
    }
    return false;
}

/* Check if player has won */
static inline bool has_won(uint32_t board) {
    return check_same_size_win(board, 0) ||
           check_same_size_win(board, 1) ||
           check_same_size_win(board, 2) ||
           check_nested_win(board) ||
           check_sequence_win(board);
}

/* Generate valid moves, return count */
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

/* Apply move to state */
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

/* Buffer for building JSON */
#define BUF_SIZE (1024 * 1024 * 64)  /* 64MB buffer per worker */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} Buffer;

static void buf_init(Buffer *b) {
    b->cap = BUF_SIZE;
    b->data = malloc(b->cap);
    b->len = 0;
}

static void buf_free(Buffer *b) {
    free(b->data);
}

static void buf_ensure(Buffer *b, size_t need) {
    if (b->len + need > b->cap) {
        b->cap *= 2;
        b->data = realloc(b->data, b->cap);
    }
}

static void buf_append(Buffer *b, const char *s, size_t len) {
    buf_ensure(b, len);
    memcpy(b->data + b->len, s, len);
    b->len += len;
}

static void buf_printf(Buffer *b, const char *fmt, ...) {
    char tmp[256];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(tmp, sizeof(tmp), fmt, args);
    va_end(args);
    buf_append(b, tmp, len);
}

/* Recursive tree exploration with streaming output */
static void explore(State *s, int depth, Buffer *buf, Stats *stats, bool first_child, Move last_move) {
    stats->nodes++;

    char move_str[4];
    snprintf(move_str, sizeof(move_str), "%c%s", SIZE_CHARS[last_move.size], POS_NAMES[last_move.pos]);

    /* Comma before non-first children */
    if (!first_child) {
        buf_append(buf, ",", 1);
    }

    /* Check for Blue win (previous player was Blue if current turn is Red) */
    if (s->turn == PLAYER_RED && has_won(s->blue)) {
        stats->blue_wins++;
        buf_printf(buf, "{\"m\":\"%s\",\"r\":\"B\"}", move_str);
        return;
    }

    /* Check for Red win */
    if (s->turn == PLAYER_BLUE && has_won(s->red)) {
        stats->red_wins++;
        buf_printf(buf, "{\"m\":\"%s\",\"r\":\"R\"}", move_str);
        return;
    }

    /* Generate moves */
    Move moves[27];
    int num_moves = generate_moves(s, moves);

    /* Draw - no moves left */
    if (num_moves == 0) {
        stats->draws++;
        buf_printf(buf, "{\"m\":\"%s\",\"r\":\"D\"}", move_str);
        return;
    }

    /* Depth limit */
    if (g_max_depth > 0 && depth >= g_max_depth) {
        stats->depth_limit++;
        buf_printf(buf, "{\"m\":\"%s\",\"r\":\"L\"}", move_str);
        return;
    }

    /* Internal node */
    buf_printf(buf, "{\"m\":\"%s\",\"c\":[", move_str);

    for (int i = 0; i < num_moves; i++) {
        State child = *s;
        apply_move(&child, moves[i]);
        explore(&child, depth + 1, buf, stats, i == 0, moves[i]);
    }

    buf_append(buf, "]}\n", 3);
}

/* Worker thread data */
typedef struct {
    int move_idx;
    Move move;
    State state;
    Buffer buf;
    Stats stats;
} WorkerData;

static void *worker_thread(void *arg) {
    WorkerData *w = (WorkerData *)arg;

    buf_init(&w->buf);
    memset(&w->stats, 0, sizeof(Stats));

    /* Start exploration from depth 1 */
    explore(&w->state, 1, &w->buf, &w->stats, true, w->move);

    return NULL;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -d, --depth N     Limit depth to N moves\n");
    fprintf(stderr, "  -p, --parallel N  Use N parallel workers (default: 1)\n");
    fprintf(stderr, "  -o, --output FILE Write JSON to FILE (default: stdout)\n");
    fprintf(stderr, "  -h, --help        Show this help\n");
}

int main(int argc, char **argv) {
    const char *output_file = NULL;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--depth") == 0) {
            if (++i < argc) g_max_depth = atoi(argv[i]);
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--parallel") == 0) {
            if (++i < argc) g_num_workers = atoi(argv[i]);
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i < argc) output_file = argv[i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "OTRIO GAME TREE EXPLORER (C)\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "Output: %s\n", output_file ? output_file : "stdout");
    fprintf(stderr, "Workers: %d\n", g_num_workers);
    if (g_max_depth > 0) fprintf(stderr, "Max depth: %d\n", g_max_depth);
    fprintf(stderr, "\n");

    /* Open output file */
    g_output = output_file ? fopen(output_file, "w") : stdout;
    if (!g_output) {
        perror("Failed to open output file");
        return 1;
    }

    /* Generate opening moves */
    State initial;
    state_init(&initial);
    Move opening_moves[27];
    int num_opening = generate_moves(&initial, opening_moves);

    fprintf(stderr, "Parallelizing %d opening moves across %d workers...\n\n", num_opening, g_num_workers);

    clock_t start = clock();

    /* Prepare worker data */
    WorkerData *workers = malloc(num_opening * sizeof(WorkerData));
    for (int i = 0; i < num_opening; i++) {
        workers[i].move_idx = i;
        workers[i].move = opening_moves[i];
        workers[i].state = initial;
        apply_move(&workers[i].state, opening_moves[i]);
    }

    /* Run workers in batches */
    pthread_t *threads = malloc(g_num_workers * sizeof(pthread_t));
    int completed = 0;

    while (completed < num_opening) {
        int batch_size = (num_opening - completed < g_num_workers) ?
                         (num_opening - completed) : g_num_workers;

        /* Launch batch */
        for (int i = 0; i < batch_size; i++) {
            pthread_create(&threads[i], NULL, worker_thread, &workers[completed + i]);
        }

        /* Wait for batch */
        for (int i = 0; i < batch_size; i++) {
            pthread_join(threads[i], NULL);
            WorkerData *w = &workers[completed + i];
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            fprintf(stderr, "  %c%s done: nodes=%lu B=%lu R=%lu D=%lu L=%lu [%.1fs]\n",
                    SIZE_CHARS[w->move.size], POS_NAMES[w->move.pos],
                    w->stats.nodes, w->stats.blue_wins, w->stats.red_wins,
                    w->stats.draws, w->stats.depth_limit, elapsed);
        }

        completed += batch_size;
    }

    free(threads);

    /* Merge results */
    fprintf(stderr, "\nMerging results...\n");

    Stats total = {0};
    fprintf(g_output, "{\"tree\":{\"c\":[");

    for (int i = 0; i < num_opening; i++) {
        if (i > 0) fprintf(g_output, ",");
        fwrite(workers[i].buf.data, 1, workers[i].buf.len, g_output);

        total.nodes += workers[i].stats.nodes;
        total.blue_wins += workers[i].stats.blue_wins;
        total.red_wins += workers[i].stats.red_wins;
        total.draws += workers[i].stats.draws;
        total.depth_limit += workers[i].stats.depth_limit;

        buf_free(&workers[i].buf);
    }

    fprintf(g_output, "]}\n,\"stats\":{\"nodes\":%lu,\"B\":%lu,\"R\":%lu,\"D\":%lu,\"L\":%lu}}\n",
            total.nodes, total.blue_wins, total.red_wins, total.draws, total.depth_limit);

    if (output_file) fclose(g_output);
    free(workers);

    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "Nodes: %lu\n", total.nodes);
    fprintf(stderr, "BLUE wins: %lu\n", total.blue_wins);
    fprintf(stderr, "RED wins: %lu\n", total.red_wins);
    fprintf(stderr, "Draws: %lu\n", total.draws);
    if (g_max_depth > 0) fprintf(stderr, "Depth limit: %lu\n", total.depth_limit);
    fprintf(stderr, "Time: %.1fs\n", elapsed);
    if (output_file) fprintf(stderr, "\nWrote to %s\n", output_file);

    return 0;
}
