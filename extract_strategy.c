/*
 * Extract Winning Strategy from Otrio Game Tree
 *
 * Loads the JSON tree into memory and computes the winning strategy subtree
 * using minimax: BLUE wins if there exists a move where all RED responses lose.
 *
 * Compile: gcc -O3 -o extract_strategy extract_strategy.c
 * Usage:   ./extract_strategy depth7.json -o strategy.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

/* Node structure - compact representation */
typedef struct Node {
    char move[4];           /* Move string (e.g., "SA1") */
    char result;            /* 'B', 'R', 'D', 'L', or 0 for internal */
    uint32_t num_children;
    struct Node *children;
    int8_t winning;         /* 1=BLUE wins, -1=RED wins, 0=unknown/limit */
} Node;

/* Statistics */
static uint64_t g_nodes_parsed = 0;
static uint64_t g_nodes_winning = 0;
static uint64_t g_blue_wins = 0;
static uint64_t g_red_wins = 0;
static uint64_t g_depth_limit = 0;

/* Skip whitespace */
static inline void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

/* Parse a JSON string (simple - no escapes) */
static void parse_string(const char **p, char *out, int maxlen) {
    skip_ws(p);
    if (**p != '"') { *out = 0; return; }
    (*p)++;
    int i = 0;
    while (**p && **p != '"' && i < maxlen - 1) {
        out[i++] = *(*p)++;
    }
    out[i] = 0;
    if (**p == '"') (*p)++;
}

/* Parse a node recursively */
static Node *parse_node(const char **p) {
    skip_ws(p);
    if (**p != '{') return NULL;
    (*p)++;

    Node *node = calloc(1, sizeof(Node));
    g_nodes_parsed++;

    if (g_nodes_parsed % 10000000 == 0) {
        fprintf(stderr, "  Parsed %lu M nodes...\n", g_nodes_parsed / 1000000);
    }

    while (**p && **p != '}') {
        skip_ws(p);
        if (**p == '"') {
            char key[16];
            parse_string(p, key, sizeof(key));
            skip_ws(p);
            if (**p == ':') (*p)++;
            skip_ws(p);

            if (strcmp(key, "m") == 0) {
                parse_string(p, node->move, sizeof(node->move));
            } else if (strcmp(key, "r") == 0) {
                char result[4];
                parse_string(p, result, sizeof(result));
                node->result = result[0];
                if (node->result == 'B') g_blue_wins++;
                else if (node->result == 'R') g_red_wins++;
                else if (node->result == 'L') g_depth_limit++;
            } else if (strcmp(key, "c") == 0) {
                /* Parse children array */
                skip_ws(p);
                if (**p == '[') {
                    (*p)++;
                    /* Count children first (rough estimate) */
                    int capacity = 32;
                    node->children = malloc(capacity * sizeof(Node));
                    node->num_children = 0;

                    while (**p && **p != ']') {
                        skip_ws(p);
                        if (**p == '{') {
                            if (node->num_children >= capacity) {
                                capacity *= 2;
                                node->children = realloc(node->children, capacity * sizeof(Node));
                            }
                            Node *child = parse_node(p);
                            if (child) {
                                node->children[node->num_children++] = *child;
                                free(child);
                            }
                        }
                        skip_ws(p);
                        if (**p == ',') (*p)++;
                    }
                    if (**p == ']') (*p)++;
                }
            }
        }
        skip_ws(p);
        if (**p == ',') (*p)++;
    }
    if (**p == '}') (*p)++;

    return node;
}

/*
 * Compute winning status using minimax.
 * depth 0 = root (no move yet, BLUE to play)
 * depth 1 = after BLUE's 1st move, RED to play
 * etc.
 *
 * Returns: 1 if BLUE wins, -1 if RED wins, 0 if unknown (depth limit)
 */
static int compute_winning(Node *node, int depth) {
    /* Leaf node */
    if (node->result) {
        if (node->result == 'B') {
            node->winning = 1;
            return 1;
        } else if (node->result == 'R') {
            node->winning = -1;
            return -1;
        } else {
            /* Draw or depth limit - unknown */
            node->winning = 0;
            return 0;
        }
    }

    /* Internal node */
    if (node->num_children == 0) {
        node->winning = 0;
        return 0;
    }

    bool is_blue_turn = (depth % 2 == 0);

    if (is_blue_turn) {
        /* BLUE's turn: BLUE wins if ANY child is winning for BLUE */
        int best = -1;
        for (uint32_t i = 0; i < node->num_children; i++) {
            int child_result = compute_winning(&node->children[i], depth + 1);
            if (child_result > best) best = child_result;
            if (best == 1) break;  /* Found a winning move */
        }
        node->winning = best;
        return best;
    } else {
        /* RED's turn: BLUE wins only if ALL children are winning for BLUE */
        int worst = 1;
        for (uint32_t i = 0; i < node->num_children; i++) {
            int child_result = compute_winning(&node->children[i], depth + 1);
            if (child_result < worst) worst = child_result;
            if (worst == -1) break;  /* RED has an escape */
        }
        node->winning = worst;
        return worst;
    }
}

/* Count nodes in winning subtree */
static void count_winning_nodes(Node *node, int depth) {
    if (node->winning != 1) return;

    g_nodes_winning++;

    if (node->result) return;

    bool is_blue_turn = (depth % 2 == 0);

    if (is_blue_turn) {
        /* BLUE's turn: only count ONE winning child (the strategy) */
        for (uint32_t i = 0; i < node->num_children; i++) {
            if (node->children[i].winning == 1) {
                count_winning_nodes(&node->children[i], depth + 1);
                break;
            }
        }
    } else {
        /* RED's turn: count ALL children (must handle all responses) */
        for (uint32_t i = 0; i < node->num_children; i++) {
            count_winning_nodes(&node->children[i], depth + 1);
        }
    }
}

/* Output winning strategy to file */
static void output_strategy(FILE *f, Node *node, int depth, bool first) {
    if (node->winning != 1) return;

    if (!first) fprintf(f, ",");

    if (node->result) {
        /* Leaf - BLUE win */
        fprintf(f, "{\"m\":\"%s\",\"r\":\"B\"}", node->move);
        return;
    }

    bool is_blue_turn = (depth % 2 == 0);

    if (node->move[0]) {
        fprintf(f, "{\"m\":\"%s\",\"c\":[", node->move);
    } else {
        fprintf(f, "{\"c\":[");
    }

    if (is_blue_turn) {
        /* BLUE's turn: output ONE winning child */
        for (uint32_t i = 0; i < node->num_children; i++) {
            if (node->children[i].winning == 1) {
                output_strategy(f, &node->children[i], depth + 1, true);
                break;
            }
        }
    } else {
        /* RED's turn: output ALL children */
        bool child_first = true;
        for (uint32_t i = 0; i < node->num_children; i++) {
            if (node->children[i].winning == 1) {
                output_strategy(f, &node->children[i], depth + 1, child_first);
                child_first = false;
            }
        }
    }

    fprintf(f, "]}");
}

/* Output all winning paths as text */
static void output_paths(FILE *f, Node *node, int depth, char *path, int path_len) {
    if (node->winning != 1) return;

    int new_len = path_len;
    if (node->move[0]) {
        if (path_len > 0) {
            path[path_len++] = ' ';
            path[path_len++] = '-';
            path[path_len++] = '>';
            path[path_len++] = ' ';
        }
        strcpy(path + path_len, node->move);
        new_len = path_len + strlen(node->move);
    }

    if (node->result == 'B') {
        /* Leaf - print path */
        path[new_len] = 0;
        fprintf(f, "%s\n", path);
        return;
    }

    bool is_blue_turn = (depth % 2 == 0);

    if (is_blue_turn) {
        /* BLUE's turn: follow ONE winning child */
        for (uint32_t i = 0; i < node->num_children; i++) {
            if (node->children[i].winning == 1) {
                output_paths(f, &node->children[i], depth + 1, path, new_len);
                break;
            }
        }
    } else {
        /* RED's turn: follow ALL children */
        for (uint32_t i = 0; i < node->num_children; i++) {
            if (node->children[i].winning == 1) {
                output_paths(f, &node->children[i], depth + 1, path, new_len);
            }
        }
    }
}

static void free_tree(Node *node) {
    if (!node) return;
    for (uint32_t i = 0; i < node->num_children; i++) {
        free_tree(&node->children[i]);
    }
    free(node->children);
}

int main(int argc, char **argv) {
    const char *input_file = NULL;
    const char *output_file = NULL;
    const char *paths_file = NULL;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            paths_file = argv[++i];
        } else if (argv[i][0] != '-') {
            input_file = argv[i];
        }
    }

    if (!input_file) {
        fprintf(stderr, "Usage: %s input.json [-o strategy.json] [-p paths.txt]\n", argv[0]);
        fprintf(stderr, "  -o FILE  Output winning strategy as JSON\n");
        fprintf(stderr, "  -p FILE  Output winning paths as text\n");
        return 1;
    }

    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "OTRIO WINNING STRATEGY EXTRACTOR\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "Input: %s\n", input_file);
    if (output_file) fprintf(stderr, "Output JSON: %s\n", output_file);
    if (paths_file) fprintf(stderr, "Output paths: %s\n", paths_file);
    fprintf(stderr, "\n");

    /* Load file into memory */
    fprintf(stderr, "Loading file into memory...\n");
    FILE *f = fopen(input_file, "rb");
    if (!f) {
        perror("Failed to open input file");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    fprintf(stderr, "  File size: %.2f GB\n", file_size / 1e9);

    char *data = malloc(file_size + 1);
    if (!data) {
        fprintf(stderr, "Failed to allocate %.2f GB\n", file_size / 1e9);
        return 1;
    }

    clock_t start = clock();
    size_t read = fread(data, 1, file_size, f);
    fclose(f);
    data[read] = 0;

    double load_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "  Loaded in %.1f seconds\n", load_time);
    fprintf(stderr, "\n");

    /* Parse JSON */
    fprintf(stderr, "Parsing JSON...\n");
    start = clock();

    const char *p = data;
    skip_ws(&p);

    /* Find "tree": */
    while (*p && strncmp(p, "\"tree\"", 6) != 0) p++;
    if (*p) {
        p += 6;
        skip_ws(&p);
        if (*p == ':') p++;
    }

    Node *root = parse_node(&p);

    double parse_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "  Parsed %lu nodes in %.1f seconds\n", g_nodes_parsed, parse_time);
    fprintf(stderr, "  BLUE wins: %lu\n", g_blue_wins);
    fprintf(stderr, "  RED wins: %lu\n", g_red_wins);
    fprintf(stderr, "  Depth limit: %lu\n", g_depth_limit);
    fprintf(stderr, "\n");

    /* Free the raw JSON data */
    free(data);

    /* Compute winning status */
    fprintf(stderr, "Computing winning strategy (minimax)...\n");
    start = clock();

    int result = compute_winning(root, 0);

    double compute_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "  Done in %.1f seconds\n", compute_time);
    fprintf(stderr, "\n");

    fprintf(stderr, "============================================================\n");
    if (result == 1) {
        fprintf(stderr, "RESULT: BLUE HAS A WINNING STRATEGY!\n");

        /* Count winning nodes */
        count_winning_nodes(root, 0);
        fprintf(stderr, "  Winning subtree: %lu nodes\n", g_nodes_winning);

        /* Output strategy JSON */
        if (output_file) {
            fprintf(stderr, "\nWriting strategy to %s...\n", output_file);
            FILE *out = fopen(output_file, "w");
            if (out) {
                fprintf(out, "{\"tree\":");
                output_strategy(out, root, 0, true);
                fprintf(out, "}\n");
                fclose(out);
                fprintf(stderr, "  Done!\n");
            }
        }

        /* Output paths */
        if (paths_file) {
            fprintf(stderr, "\nWriting winning paths to %s...\n", paths_file);
            FILE *out = fopen(paths_file, "w");
            if (out) {
                char path[1024];
                output_paths(out, root, 0, path, 0);
                fclose(out);
                fprintf(stderr, "  Done!\n");
            }
        }
    } else if (result == -1) {
        fprintf(stderr, "RESULT: RED has a winning strategy (BLUE cannot force a win)\n");
    } else {
        fprintf(stderr, "RESULT: No guaranteed winner at this depth\n");
        fprintf(stderr, "  (Need to explore deeper to determine forced win)\n");
    }
    fprintf(stderr, "============================================================\n");

    free_tree(root);
    free(root);

    return 0;
}
