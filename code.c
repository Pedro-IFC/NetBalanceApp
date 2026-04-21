#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <ctype.h>
#include <omp.h> 

typedef struct {
    double f_linear;
    double f_path;
} MultiFitness;

typedef struct {
    double avg_path;
    double diameter;
    double score;
} PathMetrics;

typedef struct {
    int u;
    int v;
    double value;
} EdgeInfo;

typedef struct {
    int N;
    int POP_SIZE;
    int GEN;
    double MU_TAX_BASE;
    int TOURNAMENT_SIZE;
    int EVAL_MATRICES;
    int EVAL_LOOPS;
    int REGEN_INTERVAL;
    double W_LINEAR;       // Peso para o solver linear
    char fitness_model[32];
} Config;

Config cfg;

double **min_matrix;
double **max_matrix;
double **value_matrix;
double ***evaluation_matrices;
int **initial_positions;
int *max_connections_per_node;
double *b_vector;
EdgeInfo *edge_index;
int edge_index_count;
int use_range_matrices;

int dominates(MultiFitness a, MultiFitness b);
int solve_gauss_fallback(double **A_in, const double *b_in, double *x_out, double **aug_buffer);
double drand_3_casas(void);
void solve_analitica_fallback(double c, const double *b, double *x);

// --- Funções de Alocação ---
int** alocar_matriz_int(int rows, int cols) {
    int **mat = (int**)malloc(rows * sizeof(int*));
    for(int i = 0; i < rows; i++) mat[i] = (int*)calloc(cols, sizeof(int)); 
    return mat;
}

double** alocar_matriz_double(int rows, int cols) {
    double **mat = (double**)malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) mat[i] = (double*)calloc(cols, sizeof(double));
    return mat;
}

int*** alocar_populacao(int pop_size, int n) {
    int ***pop = (int***)malloc(pop_size * sizeof(int**));
    for(int k = 0; k < pop_size; k++) pop[k] = alocar_matriz_int(n, n);
    return pop;
}

double*** alocar_matrizes_avaliacao(int qtd, int n) {
    double ***mat = (double***)malloc(qtd * sizeof(double**));
    for(int k = 0; k < qtd; k++) mat[k] = alocar_matriz_double(n, n);
    return mat;
}

int* alocar_vetor_int(int n) { return (int*)calloc(n, sizeof(int)); }
double* alocar_vetor_double(int n) { return (double*)calloc(n, sizeof(double)); }

// --- Funções de Liberação ---
void liberar_matriz_int(int **mat, int rows) {
    for(int i=0; i<rows; i++) free(mat[i]);
    free(mat);
}
void liberar_matriz_double(double **mat, int rows) {
    for(int i=0; i<rows; i++) free(mat[i]);
    free(mat);
}
void liberar_populacao(int ***pop, int pop_size, int rows) {
    for(int k=0; k<pop_size; k++) liberar_matriz_int(pop[k], rows);
    free(pop);
}
void liberar_matrizes_avaliacao(double ***mat, int qtd, int rows) {
    for(int k=0; k<qtd; k++) liberar_matriz_double(mat[k], rows);
    free(mat);
}

// --- Utilitários ---
void trim(char *s) {
    char *p = s;
    int l = (int)strlen(p);
    while (l > 0 && isspace((unsigned char)p[l - 1])) p[--l] = 0;
    while (*p && isspace((unsigned char)*p)) ++p, --l;
    memmove(s, p, l + 1);
}

void to_lower_str(char *s) {
    while (*s) {
        *s = (char)tolower((unsigned char)*s);
        ++s;
    }
}

void copy_str_trunc(char *dst, size_t dst_size, const char *src) {
    if (dst_size == 0) return;
    strncpy(dst, src, dst_size - 1);
    dst[dst_size - 1] = '\0';
}

void force_zero_diagonal_int(int **mat) {
    for (int i = 0; i < cfg.N; ++i) mat[i][i] = 0;
}

void force_zero_diagonal_double(double **mat) {
    for (int i = 0; i < cfg.N; ++i) mat[i][i] = 0.0;
}

int value_edge_available(int i, int j) {
    if (i == j) return 0;
    if (use_range_matrices) return fabs(max_matrix[i][j]) > 1e-12;
    return fabs(value_matrix[i][j]) > 1e-12;
}

void build_edge_index(void) {
    int max_edges = (cfg.N * (cfg.N - 1)) / 2;
    edge_index = (EdgeInfo*)malloc(max_edges * sizeof(EdgeInfo));
    edge_index_count = 0;

    for (int i = 0; i < cfg.N; ++i) {
        for (int j = i + 1; j < cfg.N; ++j) {
            if (value_edge_available(i, j)) {
                edge_index[edge_index_count++] = (EdgeInfo){i, j, value_matrix[i][j]};
            }
        }
    }
}

PathMetrics calculate_path_metrics(int **adj) {
    PathMetrics metrics = {0.0, 0.0, 0.0};
    if (cfg.N <= 1) {
        metrics.score = 2.0;
        return metrics;
    }

    int dist[cfg.N];
    int queue[cfg.N];
    double total_dist = 0.0;
    int reachable_pairs = 0;
    int max_possible_pairs = (cfg.N * (cfg.N - 1)) / 2;
    int diameter = 0;

    for (int s = 0; s < cfg.N; ++s) {
        for (int i = 0; i < cfg.N; ++i) dist[i] = -1;
        int head = 0, tail = 0;
        dist[s] = 0;
        queue[tail++] = s;

        while (head < tail) {
            int u = queue[head++];
            for (int v = 0; v < cfg.N; ++v) {
                // Checa adjacência simétrica
                if (u != v && adj[u][v] && dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    queue[tail++] = v;
                }
            }
        }

        for (int t = s + 1; t < cfg.N; ++t) {
            if (dist[t] > 0) { // Apenas se houver caminho
                total_dist += dist[t];
                reachable_pairs++;
                if (dist[t] > diameter) diameter = dist[t];
            }
        }
    }

    // Média apenas dos caminhos que existem
    metrics.avg_path = (reachable_pairs > 0) ? (total_dist / reachable_pairs) : (double)cfg.N;
    metrics.diameter = (double)diameter;

    // Fator de conectividade: Penaliza se o grafo não for totalmente conexo
    // reachable_pairs / max_possible_pairs varia de 0 a 1
    double connectivity_factor = (double)reachable_pairs / max_possible_pairs;

    // O score agora considera a eficiência do caminho E quão conexo o grafo está
    // Se não houver caminhos, o score será próximo de 0.
    metrics.score = connectivity_factor * ((1.0 / (1.0 + metrics.avg_path)) + (1.0 / (1.0 + metrics.diameter)));
    
    return metrics;
}
double infer_analitica_c(int **positions, double **matrix_values) {
    if (cfg.N <= 1) return 0.0;

    double sum = 0.0;
    int count = 0;

    for (int e = 0; e < edge_index_count; ++e) {
        EdgeInfo edge = edge_index[e];
        if (positions[edge.u][edge.v]) {
            sum += matrix_values[edge.u][edge.v];
            count++;
        }
    }

    if (count == 0) return 0.0;
    return sum / count;
}

double calculate_linear_score_analitica(int **positions, double **matrix_values, double *x_buffer) {
    double c = infer_analitica_c(positions, matrix_values);
    if (fabs(c) <= 1e-12) return 0.0;

    double total_flow = 0.0;
    solve_analitica_fallback(c, b_vector, x_buffer);
    for (int i = 0; i < cfg.N; ++i) {
        total_flow += x_buffer[i];
    }

    return total_flow;
}

double calculate_linear_score_gauss(int **positions, double **matrix_values, double **buf_A, double *buf_sol, double **buf_aug) {
    for (int i = 0; i < cfg.N; ++i) {
        memset(buf_A[i], 0, cfg.N * sizeof(double));
    }
    for (int e = 0; e < edge_index_count; ++e) {
        EdgeInfo edge = edge_index[e];
        if (positions[edge.u][edge.v]) {
            double value = matrix_values[edge.u][edge.v];
            buf_A[edge.u][edge.v] = value;
            buf_A[edge.v][edge.u] = value;
        }
    }

    memset(buf_sol, 0, cfg.N * sizeof(double));
    if (!solve_gauss_fallback(buf_A, b_vector, buf_sol, buf_aug)) {
        return 0.0;
    }

    double total_abs = 0.0;
    for (int i = 0; i < cfg.N; ++i) total_abs += fabs(buf_sol[i]);
    if (total_abs <= 1e-12) return 0.0;
    return 1.0 / total_abs;
}

double sample_matrix_value(double minv, double maxv) {
    if (maxv < minv) {
        double tmp = minv;
        minv = maxv;
        maxv = tmp;
    }
    if (maxv <= 1e-12) return 0.0;
    if (fabs(maxv - minv) <= 1e-12) return maxv;

    double step = 10.0;
    if ((maxv - minv) < step) {
        return minv + (maxv - minv) * drand_3_casas();
    }

    int steps = (int)((maxv - minv) / step) + 1;
    return minv + step * (rand() % steps);
}

void build_reference_value_matrix_from_ranges(void) {
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = 0; j < cfg.N; ++j) {
            double minv = min_matrix[i][j];
            double maxv = max_matrix[i][j];
            if (maxv < minv) {
                double tmp = minv;
                minv = maxv;
                maxv = tmp;
            }
            value_matrix[i][j] = 0.5 * (minv + maxv);
        }
    }
    force_zero_diagonal_double(value_matrix);
}

void generate_tester(double **tester) {
    for (int i = 0; i < cfg.N; ++i) {
        tester[i][i] = 0.0;
        for (int j = i + 1; j < cfg.N; ++j) {
            double value = sample_matrix_value(min_matrix[i][j], max_matrix[i][j]);
            tester[i][j] = tester[j][i] = value;
        }
    }
}

void regenerate_evaluation_matrices(void) {
    if (!use_range_matrices || evaluation_matrices == NULL) return;
    for (int t = 0; t < cfg.EVAL_MATRICES; ++t) {
        generate_tester(evaluation_matrices[t]);
    }
}


void carregar_setup(const char *arquivo) {
    FILE *f = fopen(arquivo, "r");
    if (!f) { perror("Erro ao abrir setup.temp"); exit(1); }

    char linha[512];
    char chave[100], valor[100];

    // Valores padrão caso não existam no arquivo
    cfg.EVAL_MATRICES = 20;
    cfg.EVAL_LOOPS = 20;
    cfg.REGEN_INTERVAL = 20;
    cfg.W_LINEAR = 1.0;
    copy_str_trunc(cfg.fitness_model, sizeof(cfg.fitness_model), "linear");

    while (fgets(linha, sizeof(linha), f)) {
        if (linha[0] == '#' || strlen(linha) < 2) continue;
        if (sscanf(linha, "%[^=]=%s", chave, valor) == 2) {
            trim(chave);
            trim(valor);
            if (strcmp(chave, "N") == 0) cfg.N = atoi(valor);
            else if (strcmp(chave, "POP_SIZE") == 0) cfg.POP_SIZE = atoi(valor);
            else if (strcmp(chave, "GEN") == 0) cfg.GEN = atoi(valor);
            else if (strcmp(chave, "MU_TAX_BASE") == 0) cfg.MU_TAX_BASE = atof(valor);
            else if (strcmp(chave, "TOURNAMENT_SIZE") == 0) cfg.TOURNAMENT_SIZE = atoi(valor);
            else if (strcmp(chave, "EVAL_MATRICES") == 0) cfg.EVAL_MATRICES = atoi(valor);
            else if (strcmp(chave, "EVAL_LOOPS") == 0) cfg.EVAL_LOOPS = atoi(valor);
            else if (strcmp(chave, "REGEN_INTERVAL") == 0) cfg.REGEN_INTERVAL = atoi(valor);
            else if (strcmp(chave, "W_LINEAR") == 0) cfg.W_LINEAR = atof(valor);
            else if (strcmp(chave, "fitness_model") == 0) {
                copy_str_trunc(cfg.fitness_model, sizeof(cfg.fitness_model), valor);
                to_lower_str(cfg.fitness_model);
            }
        }
    }
    
    if (cfg.N <= 0) { fprintf(stderr, "Erro: N invalido.\n"); exit(1); }

    min_matrix = alocar_matriz_double(cfg.N, cfg.N);
    max_matrix = alocar_matriz_double(cfg.N, cfg.N);
    value_matrix = alocar_matriz_double(cfg.N, cfg.N);
    initial_positions = alocar_matriz_int(cfg.N, cfg.N);
    max_connections_per_node = alocar_vetor_int(cfg.N);
    b_vector = alocar_vetor_double(cfg.N);
    evaluation_matrices = NULL;
    int loaded_min_matrix = 0;
    int loaded_max_matrix = 0;
    int loaded_value_matrix = 0;

    rewind(f);
    char header[100];
    
    while (fscanf(f, "%s", header) != EOF) {
        if (strcmp(header, "[MIN_MATRIX]") == 0) {
            for(int i=0; i<cfg.N; i++) for(int j=0; j<cfg.N; j++) fscanf(f, "%lf", &min_matrix[i][j]);
            loaded_min_matrix = 1;
        }
        else if (strcmp(header, "[MAX_MATRIX]") == 0) {
            for(int i=0; i<cfg.N; i++) for(int j=0; j<cfg.N; j++) fscanf(f, "%lf", &max_matrix[i][j]);
            loaded_max_matrix = 1;
        }
        else if (strcmp(header, "[VALUE_MATRIX]") == 0) {
            for(int i=0; i<cfg.N; i++) for(int j=0; j<cfg.N; j++) fscanf(f, "%lf", &value_matrix[i][j]);
            loaded_value_matrix = 1;
        }
        else if (strcmp(header, "[INITIAL_POSITIONS]") == 0) {
            for(int i=0; i<cfg.N; i++) for(int j=0; j<cfg.N; j++) fscanf(f, "%d", &initial_positions[i][j]);
        }
        else if (strcmp(header, "[MAX_CONNECTIONS]") == 0) {
            for(int i=0; i<cfg.N; i++) fscanf(f, "%d", &max_connections_per_node[i]);
        }
        else if (strcmp(header, "[B_VECTOR]") == 0) {
            for(int i=0; i<cfg.N; i++) fscanf(f, "%lf", &b_vector[i]);
        }
    }
    fclose(f);

    force_zero_diagonal_double(min_matrix);
    force_zero_diagonal_double(max_matrix);
    force_zero_diagonal_double(value_matrix);
    force_zero_diagonal_int(initial_positions);
    use_range_matrices = loaded_min_matrix && loaded_max_matrix;
    if (!loaded_value_matrix && use_range_matrices) {
        build_reference_value_matrix_from_ranges();
    }
    if (!loaded_value_matrix && !use_range_matrices) {
        fprintf(stderr, "Erro: forneca [VALUE_MATRIX] ou [MIN_MATRIX]/[MAX_MATRIX].\n");
        exit(1);
    }
    if (cfg.EVAL_MATRICES <= 0) cfg.EVAL_MATRICES = 1;
    if (cfg.EVAL_LOOPS <= 0) cfg.EVAL_LOOPS = cfg.EVAL_MATRICES;
    if (cfg.REGEN_INTERVAL <= 0) cfg.REGEN_INTERVAL = 1;
    build_edge_index();
}

void liberar_memoria_global() {
    if (evaluation_matrices != NULL) liberar_matrizes_avaliacao(evaluation_matrices, cfg.EVAL_MATRICES, cfg.N);
    free(edge_index);
    liberar_matriz_double(min_matrix, cfg.N);
    liberar_matriz_double(max_matrix, cfg.N);
    liberar_matriz_double(value_matrix, cfg.N);
    liberar_matriz_int(initial_positions, cfg.N);
    free(max_connections_per_node);
    free(b_vector);
}

// --- Solver Linear ---
int solve_gauss_fallback(double **A_in, const double *b_in, double *x_out, double **aug_buffer) {
    for (int i=0;i<cfg.N;++i) {
        for (int j=0;j<cfg.N;++j) aug_buffer[i][j] = A_in[i][j];
        aug_buffer[i][cfg.N] = b_in[i];
    }
    
    int success = 1;
    for (int i=0;i<cfg.N;++i) {
        int pivot=i; 
        double maxv=fabs(aug_buffer[i][i]);
        for (int r=i+1;r<cfg.N;++r){ 
            double av=fabs(aug_buffer[r][i]); 
            if (av>maxv){ maxv=av; pivot=r; } 
        }
        
        if (maxv < 1e-12) { success = 0; break; }
        
        if (pivot != i) {
            for (int c=i;c<=cfg.N;++c){ 
                double t=aug_buffer[i][c]; 
                aug_buffer[i][c]=aug_buffer[pivot][c]; 
                aug_buffer[pivot][c]=t; 
            }
        }
        
        double diag = aug_buffer[i][i];
        for (int k=i+1;k<cfg.N;++k){
            double f = aug_buffer[k][i] / diag;
            if (f==0.0) continue;
            for (int j=i;j<=cfg.N;++j) aug_buffer[k][j] -= f * aug_buffer[i][j];
        }
    }
    
    if (success) {
        for (int i=cfg.N-1;i>=0;--i){
            double s = aug_buffer[i][cfg.N];
            for (int j=i+1;j<cfg.N;++j) s -= aug_buffer[i][j] * x_out[j];
            x_out[i] = s / aug_buffer[i][i];
        }
    }
    return success;
}

void solve_analitica_fallback(double c, const double *b, double *x) {
    double S = 0.0;
    for (int i = 0; i < cfg.N; i++) S += b[i];

    double term = S / (cfg.N - 1.0);
    double inv_c = 1.0 / c;
    for (int i = 0; i < cfg.N; i++) x[i] = inv_c * (term - b[i]);
}

double drand_3_casas(void) {
    int r = rand() % 1001; 
    return r / 1000.0;
}

static inline void copy_positions(int **src, int **dst) {
    for(int i=0; i<cfg.N; i++) memcpy(dst[i], src[i], cfg.N * sizeof(int));
    force_zero_diagonal_int(dst);
}

void enforce_connection_limits(int **p, const int *max_limits) {
    int j_candidates[cfg.N];
    int current_conns[cfg.N];

    memset(current_conns, 0, cfg.N * sizeof(int));
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = i + 1; j < cfg.N; ++j) {
            if (!value_edge_available(i, j)) {
                p[i][j] = p[j][i] = 0;
            } else if (p[i][j]) {
                p[i][j] = p[j][i] = 1;
                current_conns[i]++;
                current_conns[j]++;
            } else {
                p[i][j] = p[j][i] = 0;
            }
        }
    }

    for (int i = 0; i < cfg.N; ++i) {
        int excess = current_conns[i] - max_limits[i];
        if (excess <= 0) continue;

        int num_candidates = 0;
        for (int j = 0; j < cfg.N; ++j) {
            if (i != j && p[i][j] == 1 && value_edge_available(i, j)) {
                j_candidates[num_candidates++] = j;
            }
        }
        for (int k = 0; k < excess; ++k) {
            if (num_candidates == 0) break;
            int rand_idx = rand() % num_candidates;
            int j_to_remove = j_candidates[rand_idx];
            p[i][j_to_remove] = p[j_to_remove][i] = 0;
            current_conns[i]--;
            current_conns[j_to_remove]--;
            j_candidates[rand_idx] = j_candidates[num_candidates - 1];
            num_candidates--;
        }
    }
    force_zero_diagonal_int(p);
}

MultiFitness fitness(int **positions, double **buf_A, double *buf_sol, double **buf_aug, const char *fitness_model) {
    double linear_score = 0.0;
    PathMetrics path_metrics = calculate_path_metrics(positions);
    int use_analitica = (fitness_model != NULL && strcmp(fitness_model, "analitica") == 0);
    int eval_count = (use_range_matrices && evaluation_matrices != NULL) ? cfg.EVAL_LOOPS : 1;

    for (int eval_idx = 0; eval_idx < eval_count; ++eval_idx) {
        double **matrix_values = value_matrix;
        if (use_range_matrices && evaluation_matrices != NULL) {
            matrix_values = evaluation_matrices[eval_idx % cfg.EVAL_MATRICES];
        }

        if (use_analitica) {
            linear_score += calculate_linear_score_analitica(positions, matrix_values, buf_sol);
        } else {
            linear_score += calculate_linear_score_gauss(positions, matrix_values, buf_A, buf_sol, buf_aug);
        }
    }

    if (eval_count > 1) linear_score /= eval_count;
    return (MultiFitness){linear_score, path_metrics.score};
}

void randomize(int **p) {
    for (int i = 0; i < cfg.N; ++i) memset(p[i], 0, cfg.N * sizeof(int));
    for (int e = 0; e < edge_index_count; ++e) {
        EdgeInfo edge = edge_index[e];
        int v = rand() & 1;
        p[edge.u][edge.v] = p[edge.v][edge.u] = v;
    }
    force_zero_diagonal_int(p);
    enforce_connection_limits(p, max_connections_per_node);
}

void mutate(int **src, int **dst, double mu) {
    copy_positions(src, dst);
    for (int e = 0; e < edge_index_count; ++e) {
        EdgeInfo edge = edge_index[e];
        if (drand_3_casas() < mu) {
            dst[edge.u][edge.v] = dst[edge.v][edge.u] = 1 - dst[edge.u][edge.v];
        }
    }
    enforce_connection_limits(dst, max_connections_per_node);
}

void cross(int **p1, int **p2, int **dst) {
    copy_positions(p1, dst);
    for (int e = 0; e < edge_index_count; ++e) {
        EdgeInfo edge = edge_index[e];
        if (drand_3_casas() < cfg.MU_TAX_BASE) {
            dst[edge.u][edge.v] = dst[edge.v][edge.u] = p2[edge.u][edge.v];
        }
    }
    enforce_connection_limits(dst, max_connections_per_node);
}

int select_parent(int pop, MultiFitness *fitnesses, int exclude_idx) {
    int iterations = cfg.TOURNAMENT_SIZE < pop ? cfg.TOURNAMENT_SIZE : pop;
    int best_idx = -1;

    for (int k = 0; k < iterations; ++k) {
        int idx = rand() % pop;
        if (idx == exclude_idx) continue;
        if (best_idx == -1 || dominates(fitnesses[idx], fitnesses[best_idx])) {
            best_idx = idx;
        }
    }

    if (best_idx != -1) return best_idx;
    do {
        best_idx = rand() % pop;
    } while (pop > 1 && best_idx == exclude_idx);

    return best_idx;
}

void save_value_matrix() {
    FILE *f = fopen("./files/value_matrix.csv", "w");
    if (!f) return;
    fprintf(f, "i,j,value\n");
    for (int i = 0; i < cfg.N; ++i) for (int j = 0; j < cfg.N; ++j) 
        fprintf(f, "%d,%d,%.10g\n", i, j, value_matrix[i][j]);
    fclose(f);
}

void save_tester_config() {
    FILE *f = fopen("./files/tester.csv", "w");
    if (!f) return;
    fprintf(f, "i,j,min_value,max_value\n");
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = 0; j < cfg.N; ++j) {
            fprintf(f, "%d,%d,%.10g,%.10g\n", i, j, min_matrix[i][j], max_matrix[i][j]);
        }
    }
    fclose(f);
}

void save_b_vector() {
    FILE *f = fopen("./files/b_vector.csv", "w");
    if (!f) return;
    fprintf(f, "index,value\n");
    for (int i = 0; i < cfg.N; ++i) fprintf(f, "%d,%.1f\n", i, b_vector[i]);
    fclose(f);
}

void save_complete_history_header(FILE *f) {
    fprintf(f, "Generation,Individual_ID,Fitness_Linear,Fitness_Path");
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = 0; j < cfg.N; ++j) {
            fprintf(f, ",Gene_%d_%d", i, j);
        }
    }
    fprintf(f, "\n");
}

void append_complete_history_generation(FILE *f, int gen, int ***population, MultiFitness *fitnesses) {
    for (int k = 0; k < cfg.POP_SIZE; ++k) {
        fprintf(f, "%d,%d,%f,%f", gen, k, fitnesses[k].f_linear, fitnesses[k].f_path);
        for (int i = 0; i < cfg.N; ++i) {
            for (int j = 0; j < cfg.N; ++j) {
                fprintf(f, ",%d", population[k][i][j]);
            }
        }
        fprintf(f, "\n");
    }
}

int dominates(MultiFitness a, MultiFitness b) {
    int melhor_ou_igual_em_tudo = (a.f_linear >= b.f_linear) && (a.f_path >= b.f_path);
    int estritamente_melhor_em_um = (a.f_linear > b.f_linear) || (a.f_path > b.f_path);
    
    return melhor_ou_igual_em_tudo && estritamente_melhor_em_um;
}
int main(void) {
    srand((unsigned)time(NULL));

    carregar_setup("./setup.temp");

    int ***population = alocar_populacao(cfg.POP_SIZE, cfg.N);
    int ***new_pop = alocar_populacao(cfg.POP_SIZE, cfg.N);
    
    // CORREÇÃO: O array de fitness agora é do tipo MultiFitness
    MultiFitness *fitnesses = (MultiFitness*)malloc(cfg.POP_SIZE * sizeof(MultiFitness));

    int **best_positions = alocar_matriz_int(cfg.N, cfg.N);
    int **child_temp = alocar_matriz_int(cfg.N, cfg.N);
    int **mutated_temp = alocar_matriz_int(cfg.N, cfg.N);

    double **buffer_A = alocar_matriz_double(cfg.N, cfg.N);
    double **buffer_aug = alocar_matriz_double(cfg.N, cfg.N + 1);
    double *buffer_sol = alocar_vetor_double(cfg.N);

    double mu = cfg.MU_TAX_BASE;
    int gens_no_improve = 0;

    if (use_range_matrices) {
        evaluation_matrices = alocar_matrizes_avaliacao(cfg.EVAL_MATRICES, cfg.N);
        regenerate_evaluation_matrices();
        save_tester_config();
    }
    save_value_matrix();
    save_b_vector();

    FILE *f_complete = fopen("./files/history_advanced_complete.csv", "w");
    if (!f_complete) {
        perror("Erro ao abrir history_advanced_complete.csv");
        return 1;
    }
    setvbuf(f_complete, NULL, _IOFBF, 1 << 20);
    save_complete_history_header(f_complete);

    copy_positions(initial_positions, population[0]);
    enforce_connection_limits(population[0], max_connections_per_node);
    for (int i=1;i<cfg.POP_SIZE;++i) randomize(population[i]);

    for (int i=0;i<cfg.POP_SIZE;++i) 
        fitnesses[i] = fitness(population[i], buffer_A, buffer_sol, buffer_aug, cfg.fitness_model);
    
    append_complete_history_generation(f_complete, 0, population, fitnesses);

    int curr_best_idx_gen0 = 0;
    for (int i=1;i<cfg.POP_SIZE;++i) {
        if (dominates(fitnesses[i], fitnesses[curr_best_idx_gen0])) {
            curr_best_idx_gen0 = i;
        }
    }
    
    MultiFitness best_fit_global = fitnesses[curr_best_idx_gen0];
    copy_positions(population[curr_best_idx_gen0], best_positions);
    
    FILE *f_best = fopen("./files/history_advanced_best_of_gen.csv","w");
    if (!f_best) {
        perror("Erro ao abrir history_advanced_best_of_gen.csv");
        fclose(f_complete);
        return 1;
    }
    setvbuf(f_best, NULL, _IOFBF, 1 << 20);
    fprintf(f_best, "Generation,GlobalBest_Linear,GlobalBest_Path,GenBest_Linear,GenBest_Path");
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = 0; j < cfg.N; ++j) {
            fprintf(f_best, ",Gene_%d_%d", i, j);
        }
    }
    fprintf(f_best, "\n");
    
    fprintf(f_best, "0,%f,%f,%f,%f",
            best_fit_global.f_linear, best_fit_global.f_path,
            best_fit_global.f_linear, best_fit_global.f_path); 
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = 0; j < cfg.N; ++j) {
            fprintf(f_best, ",%d", best_positions[i][j]);
        }
    }
    fprintf(f_best, "\n");

    clock_t start_time = clock();

    for (int gen=0; gen<cfg.GEN; ++gen) {
        printf("Dados:%d,%f\n", gen, best_fit_global.f_linear);

        if (use_range_matrices && gen > 0 && (gen % cfg.REGEN_INTERVAL) == 0) {
            regenerate_evaluation_matrices();
        }

        for (int i=0;i<cfg.POP_SIZE;++i) 
            fitnesses[i] = fitness(population[i], buffer_A, buffer_sol, buffer_aug, cfg.fitness_model);

        int curr_best_idx = 0;
        for (int i=1;i<cfg.POP_SIZE;++i) {
            if (dominates(fitnesses[i], fitnesses[curr_best_idx])) curr_best_idx = i;
        }
        
        MultiFitness best_fit_generation = fitnesses[curr_best_idx];

        double score_gen = best_fit_generation.f_linear + best_fit_generation.f_path; // Adicione multiplicadores de peso aqui se necessário
        double score_global = best_fit_global.f_linear + best_fit_global.f_path;

        if (score_gen > score_global) {
            best_fit_global = best_fit_generation;
            copy_positions(population[curr_best_idx], best_positions);
            gens_no_improve = 0;
            mu = cfg.MU_TAX_BASE;
        } else {
            gens_no_improve++;
            if (gens_no_improve % 50 == 0) mu += 0.025;
        }

        append_complete_history_generation(f_complete, gen, population, fitnesses);

        fprintf(f_best, "%d,%f,%f,%f,%f",
                gen,
                best_fit_global.f_linear, best_fit_global.f_path,
                best_fit_generation.f_linear, best_fit_generation.f_path);
        for (int i = 0; i < cfg.N; ++i) {
            for (int j = 0; j < cfg.N; ++j) {
                fprintf(f_best, ",%d", population[curr_best_idx][i][j]);
            }
        }
        fprintf(f_best, "\n");
        
        int cnt = 0, attempts = 0;
        while (cnt < cfg.POP_SIZE && attempts < 10 * cfg.POP_SIZE) {
            int p1 = select_parent(cfg.POP_SIZE, fitnesses, -1);
            int p2 = select_parent(cfg.POP_SIZE, fitnesses, p1);

            cross(population[p1], population[p2], child_temp); 
            mutate(child_temp, mutated_temp, mu);

            MultiFitness f1 = fitnesses[p1];
            MultiFitness f2 = fitnesses[p2];
            MultiFitness fc = fitness(mutated_temp, buffer_A, buffer_sol, buffer_aug, cfg.fitness_model);

            if (!dominates(f1, fc) && !dominates(f2, fc)) {
                copy_positions(mutated_temp, new_pop[cnt++]);
            }
            attempts++;
        }

        while (cnt < cfg.POP_SIZE) {
            copy_positions(best_positions, new_pop[cnt++]);
        }

        for (int i=0;i<cfg.POP_SIZE;++i) copy_positions(new_pop[i], population[i]);
        
        FILE *f_pareto = fopen("./files/pareto_front.csv", "w");
        if (f_pareto) {
            setvbuf(f_pareto, NULL, _IOFBF, 1 << 16);
            fprintf(f_pareto, "ID,Fitness_Linear,Fitness_Path\n"); 

            for (int i = 0; i < cfg.POP_SIZE; i++) {
                int is_dominated = 0;
                for (int j = 0; j < cfg.POP_SIZE; j++) {
                    if (i != j && dominates(fitnesses[j], fitnesses[i])) {
                        is_dominated = 1;
                        break;
                    }
                }
                
                if (!is_dominated) {
                    fprintf(f_pareto, "%d,%f,%f\n", i, fitnesses[i].f_linear, fitnesses[i].f_path);
                }
            }
            fclose(f_pareto);
        } else {
            perror("ERRO CRÍTICO: Não foi possível abrir ./files/pareto_front.csv para escrita!");
        }
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nMelhor global (Linear): %f\n", best_fit_global.f_linear);
    printf("Melhor global (Path): %f\n", best_fit_global.f_path);
    printf("Tempo: %.3f s\n", elapsed_time);

    // Limpeza final
    liberar_populacao(population, cfg.POP_SIZE, cfg.N);
    liberar_populacao(new_pop, cfg.POP_SIZE, cfg.N);
    liberar_matriz_int(best_positions, cfg.N);
    liberar_matriz_int(child_temp, cfg.N);
    liberar_matriz_int(mutated_temp, cfg.N);
    
    liberar_matriz_double(buffer_A, cfg.N);
    liberar_matriz_double(buffer_aug, cfg.N);
    free(buffer_sol);
    free(fitnesses);
    liberar_memoria_global();

    return 0;
}
