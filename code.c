#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <omp.h> 

typedef struct {
    int N;
    int POP_SIZE;
    int GEN;
    double MU_TAX_BASE;
    int TOURNAMENT_SIZE;
    int EVAL_MATRICES;
    int EVAL_LOOPS;
    int REGEN_INTERVAL;
} Config;

Config cfg;

int **min_matrix;
int **max_matrix;
int **initial_positions;
int *max_connections_per_node;
double *b_vector;

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
    int l = strlen(p);
    while(isspace(p[l - 1])) p[--l] = 0;
    while(*p && isspace(*p)) ++p, --l;
    memmove(s, p, l + 1);
}

void carregar_setup(const char *arquivo) {
    FILE *f = fopen(arquivo, "r");
    if (!f) { perror("Erro ao abrir setup.temp"); exit(1); }

    char linha[512];
    char chave[100], valor[100];

    while (fgets(linha, sizeof(linha), f)) {
        if (linha[0] == '#' || strlen(linha) < 2) continue;
        if (sscanf(linha, "%[^=]=%s", chave, valor) == 2) {
            trim(chave);
            if (strcmp(chave, "N") == 0) cfg.N = atoi(valor);
            else if (strcmp(chave, "POP_SIZE") == 0) cfg.POP_SIZE = atoi(valor);
            else if (strcmp(chave, "GEN") == 0) cfg.GEN = atoi(valor);
            else if (strcmp(chave, "MU_TAX_BASE") == 0) cfg.MU_TAX_BASE = atof(valor);
            else if (strcmp(chave, "TOURNAMENT_SIZE") == 0) cfg.TOURNAMENT_SIZE = atoi(valor);
            else if (strcmp(chave, "EVAL_MATRICES") == 0) cfg.EVAL_MATRICES = atoi(valor);
            else if (strcmp(chave, "EVAL_LOOPS") == 0) cfg.EVAL_LOOPS = atoi(valor);
            else if (strcmp(chave, "REGEN_INTERVAL") == 0) cfg.REGEN_INTERVAL = atoi(valor);
        }
    }
    
    if (cfg.N <= 0) { fprintf(stderr, "Erro: N invalido.\n"); exit(1); }

    min_matrix = alocar_matriz_int(cfg.N, cfg.N);
    max_matrix = alocar_matriz_int(cfg.N, cfg.N);
    initial_positions = alocar_matriz_int(cfg.N, cfg.N);
    max_connections_per_node = alocar_vetor_int(cfg.N);
    b_vector = alocar_vetor_double(cfg.N);

    rewind(f);
    char header[100];
    
    while (fscanf(f, "%s", header) != EOF) {
        if (strcmp(header, "[MIN_MATRIX]") == 0) {
            for(int i=0; i<cfg.N; i++) for(int j=0; j<cfg.N; j++) fscanf(f, "%d", &min_matrix[i][j]);
        }
        else if (strcmp(header, "[MAX_MATRIX]") == 0) {
            for(int i=0; i<cfg.N; i++) for(int j=0; j<cfg.N; j++) fscanf(f, "%d", &max_matrix[i][j]);
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
}

void liberar_memoria_global() {
    liberar_matriz_int(min_matrix, cfg.N);
    liberar_matriz_int(max_matrix, cfg.N);
    liberar_matriz_int(initial_positions, cfg.N);
    free(max_connections_per_node);
    free(b_vector);
}

// --- Solver Linear (OTIMIZADO: Recebe buffer para evitar malloc) ---
int solve_linear_fallback(double **A_in, const double *b_in, double *x_out, double **aug_buffer) {
    // Copia para o buffer pré-alocado
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
    // Não libera aug_buffer, ele é reutilizado
    return success;
}

double drand_3_casas(void) {
    int r = rand() % 1001; 
    return r / 1000.0;
}

static inline void copy_positions(int **src, int **dst) { 
    for(int i=0; i<cfg.N; i++) memcpy(dst[i], src[i], cfg.N * sizeof(int));
}

void generate_tester(double **tester) {
    for (int i=0;i<cfg.N;++i) for (int j=0;j<=i;++j) {
        int minv = min_matrix[i][j];
        int maxv = max_matrix[i][j];
        int v = 0;
        if (maxv > 0 && maxv >= minv) {
            int steps = (maxv - minv) / 10 + 1;
            v = minv + 10 * (rand() % steps);
        }
        tester[i][j] = tester[j][i] = (double)v;
    }
}

int count_node_connections(int **p, int node_i) {
    int count = 0;
    for (int j = 0; j < cfg.N; ++j) {
        if (node_i != j && p[node_i][j] == 1) count++;
    }
    return count;
}

void enforce_connection_limits(int **p, const int *max_limits) {
    int *j_candidates = alocar_vetor_int(cfg.N);
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = i + 1; j < cfg.N; ++j) {
            if (min_matrix[i][j] == max_matrix[i][j] && max_matrix[i][j] > 0) p[i][j] = p[j][i] = 1;
            else if (max_matrix[i][j] == 0) p[i][j] = p[j][i] = 0;
        }
    }
    for (int i = 0; i < cfg.N; ++i) {
        int current_conns = count_node_connections(p, i);
        int excess = current_conns - max_limits[i];
        if (excess <= 0) continue;

        int num_candidates = 0;
        for (int j = 0; j < cfg.N; ++j) {
            if (i != j && p[i][j] == 1 && (min_matrix[i][j] != max_matrix[i][j] || max_matrix[i][j] == 0)) {
                 j_candidates[num_candidates++] = j;
            }
        }
        for (int k = 0; k < excess; ++k) {
            if (num_candidates == 0) break;
            int rand_idx = rand() % num_candidates;
            int j_to_remove = j_candidates[rand_idx];
            p[i][j_to_remove] = p[j_to_remove][i] = 0;
            j_candidates[rand_idx] = j_candidates[num_candidates - 1];
            num_candidates--;
        }
    }
    free(j_candidates);
}

// --- Fitness (OTIMIZADO: Recebe buffers) ---
double fitness(int **positions, double ***testers, int num_testers, int loops, 
               double **buf_A, double *buf_sol, double **buf_aug) {
    double final_point = 0.0;
    
    for (int f=0; f<loops; ++f) {
        int idx = f % num_testers;
        
        // Monta matriz A no buffer
        for (int i=0;i<cfg.N;++i) 
            for (int j=0;j<cfg.N;++j) 
                buf_A[i][j] = positions[i][j] ? testers[idx][i][j] : 0.0;
        
        memset(buf_sol, 0, cfg.N * sizeof(double));
        
        // Passa buf_aug para o solver
        if (!solve_linear_fallback(buf_A, b_vector, buf_sol, buf_aug)) continue;
        
        double total_abs = 0.0;
        for (int i=0;i<cfg.N;++i) total_abs += fabs(buf_sol[i]);
        if (total_abs == 0.0) continue;
        final_point += 1.0 / total_abs;
    }
    
    return final_point/cfg.EVAL_MATRICES;
}

void randomize(int **p) {
    for (int i=0;i<cfg.N;++i) for (int j=0;j<=i;++j) {
        int v = rand() & 1;
        if(max_matrix[i][j] == 0) v = 0;
        p[i][j] = p[j][i] = v;
    }
    enforce_connection_limits(p, max_connections_per_node);
}

void mutate(int **src, int **dst, double mu) {
    copy_positions(src, dst);
    for (int i=0; i<cfg.N; ++i) {
        for (int j=i+1; j<cfg.N; ++j) {
            if(max_matrix[i][j] > 0 && drand_3_casas() < mu) {
                dst[i][j] = dst[j][i] = 1 - dst[i][j];
            }
        }
    }
    enforce_connection_limits(dst, max_connections_per_node);
}

void cross(int **p1, int **p2, int **dst) {
    copy_positions(p1, dst);
    for (int i=0; i<cfg.N; ++i) {
        for (int j=i+1; j<cfg.N; ++j) {
            if (drand_3_casas() < cfg.MU_TAX_BASE) dst[i][j] = dst[j][i] = p2[i][j];
        }
    }
    enforce_connection_limits(dst, max_connections_per_node);
}

int select_parent(int pop, const double *fitnesses, int exclude) {
    int best_idx = rand() % pop;
    double best_fit = fitnesses[best_idx];
    int iterations = (cfg.TOURNAMENT_SIZE < pop) ? cfg.TOURNAMENT_SIZE : pop;
    for (int k=1;k<iterations;++k) {
        int idx = rand() % pop;
        if (idx == exclude) continue;
        if (fitnesses[idx] > best_fit) { best_fit = fitnesses[idx]; best_idx = idx; }
    }
    return best_idx;
}

void save_tester_config() {
    FILE *f = fopen("tester.csv", "w");
    if (!f) return;
    fprintf(f, "i,j,min_value,max_value\n");
    for (int i = 0; i < cfg.N; ++i) for (int j = 0; j < cfg.N; ++j) 
        fprintf(f, "%d,%d,%d,%d\n", i, j, min_matrix[i][j], max_matrix[i][j]);
    fclose(f);
}

void save_b_vector() {
    FILE *f = fopen("b_vector.csv", "w");
    if (!f) return;
    fprintf(f, "index,value\n");
    for (int i = 0; i < cfg.N; ++i) fprintf(f, "%d,%.1f\n", i, b_vector[i]);
    fclose(f);
}

void save_complete_history_header(FILE *f) {
    fprintf(f, "Generation,Individual_ID,Fitness");
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = 0; j < cfg.N; ++j) {
            fprintf(f, ",Gene_%d_%d", i, j);
        }
    }
    fprintf(f, "\n");
}

// Alterado: population agora é int *** e fitnesses é double *
void append_complete_history_generation(FILE *f, int gen, int ***population, double *fitnesses) {
    for (int k = 0; k < cfg.POP_SIZE; ++k) {
        fprintf(f, "%d,%d,%f", gen, k, fitnesses[k]);
        for (int i = 0; i < cfg.N; ++i) {
            for (int j = 0; j < cfg.N; ++j) {
                fprintf(f, ",%d", population[k][i][j]);
            }
        }
        fprintf(f, "\n");
    }
}
// ==========================================
// MAIN
// ==========================================

int main(void) {
    srand((unsigned)time(NULL));

    // 1. Setup
    carregar_setup("setup.temp");

    int *min_connections_per_node = alocar_vetor_int(cfg.N);
    for (int i = 0; i < cfg.N; ++i) {
        for (int j = 0; j < cfg.N; ++j) {
            if (i != j && max_matrix[i][j] > 0 && min_matrix[i][j] > 0) min_connections_per_node[i]++;
        }
    }
    
    int has_conflict = 0;
    for (int i = 0; i < cfg.N; ++i) {
        if (min_connections_per_node[i] > max_connections_per_node[i]) {
            fprintf(stderr, "ERRO: Conflito no No %d.\n", i);
            has_conflict = 1; 
        }
    }
    if(has_conflict) return 1;

    // 2. Alocações Principais
    int ***population = alocar_populacao(cfg.POP_SIZE, cfg.N);
    int ***new_pop = alocar_populacao(cfg.POP_SIZE, cfg.N);
    double *fitnesses = alocar_vetor_double(cfg.POP_SIZE);
    double ***evaluation_matrices = alocar_matrizes_avaliacao(cfg.EVAL_MATRICES, cfg.N);
    
    // Alocação de temporários de mutação
    int **best_positions = alocar_matriz_int(cfg.N, cfg.N);
    int **child_temp = alocar_matriz_int(cfg.N, cfg.N);
    int **mutated_temp = alocar_matriz_int(cfg.N, cfg.N);

    // --- ALOCAÇÃO DOS BUFFERS OTIMIZADOS ---
    // Estes buffers evitam milhões de mallocs dentro do loop de fitness
    double **buffer_A = alocar_matriz_double(cfg.N, cfg.N);
    double **buffer_aug = alocar_matriz_double(cfg.N, cfg.N + 1);
    double *buffer_sol = alocar_vetor_double(cfg.N);

    double mu = cfg.MU_TAX_BASE;
    int gens_no_improve = 0;

    for (int t=0;t<cfg.EVAL_MATRICES;++t) generate_tester(evaluation_matrices[t]);
    
    save_tester_config();
    save_b_vector();
    // Prepara o arquivo de histórico completo
    FILE *f_complete = fopen("history_advanced_complete.csv", "w");
    if (!f_complete) {
        perror("Erro ao abrir history_advanced_complete.csv");
        return 1;
    }
    save_complete_history_header(f_complete);

    // Inicialização da População
    copy_positions(initial_positions, population[0]);
    enforce_connection_limits(population[0], max_connections_per_node);
    for (int i=1;i<cfg.POP_SIZE;++i) randomize(population[i]);

    // Passamos os buffers aqui
    for (int i=0;i<cfg.POP_SIZE;++i) 
        fitnesses[i] = fitness(population[i], evaluation_matrices, cfg.EVAL_MATRICES, cfg.EVAL_LOOPS, buffer_A, buffer_sol, buffer_aug);
    
    append_complete_history_generation(f_complete, 0, population, fitnesses);

    int curr_best_idx_gen0 = 0;
    for (int i=1;i<cfg.POP_SIZE;++i) if (fitnesses[i] > fitnesses[curr_best_idx_gen0]) curr_best_idx_gen0 = i;
    double best_fit_global = fitnesses[curr_best_idx_gen0];
    copy_positions(population[curr_best_idx_gen0], best_positions);
    
    clock_t start_time = clock();


    // LOOP PRINCIPAL
    for (int gen=0; gen<cfg.GEN; ++gen) {
        printf("Dados: %d, %f\n", gen, best_fit_global);
        if ((gen % cfg.REGEN_INTERVAL) == 0) {
            for (int t=0;t<cfg.EVAL_MATRICES;++t) generate_tester(evaluation_matrices[t]);
        }

        // Passamos os buffers aqui também
        for (int i=0;i<cfg.POP_SIZE;++i) 
            fitnesses[i] = fitness(population[i], evaluation_matrices, cfg.EVAL_MATRICES, cfg.EVAL_LOOPS, buffer_A, buffer_sol, buffer_aug);

        int curr_best_idx = 0;
        for (int i=1;i<cfg.POP_SIZE;++i) if (fitnesses[i] > fitnesses[curr_best_idx]) curr_best_idx = i;
        double best_fit_generation = fitnesses[curr_best_idx];

        if (best_fit_generation > best_fit_global) {
            best_fit_global = best_fit_generation;
            copy_positions(population[curr_best_idx], best_positions);
            gens_no_improve = 0;
            mu = cfg.MU_TAX_BASE;
        } else {
            gens_no_improve++;
            if (gens_no_improve % 50 == 0) mu += 0.025;
        }

        // **NOVA CHAMADA: Salva todos os indivíduos desta geração no histórico completo**
        append_complete_history_generation(f_complete, gen, population, fitnesses);

        int cnt = 0, attempts = 0;
        while (cnt < cfg.POP_SIZE && attempts < 10 * cfg.POP_SIZE) {
            int p1 = select_parent(cfg.POP_SIZE, fitnesses, cfg.POP_SIZE + 1);
            int p2 = select_parent(cfg.POP_SIZE, fitnesses, p1);

            cross(population[p1], population[p2], child_temp); 
            mutate(child_temp, mutated_temp, mu);

            double f1 = fitnesses[p1];
            double f2 = fitnesses[p2];
            // Buffers também na avaliação do filho
            double fc = fitness(mutated_temp, evaluation_matrices, cfg.EVAL_MATRICES, cfg.EVAL_LOOPS, buffer_A, buffer_sol, buffer_aug);

            if (fc > f1 && fc > f2) {
                copy_positions(mutated_temp, new_pop[cnt++]);
            }
            attempts++;
        }

        while (cnt < cfg.POP_SIZE) {
            copy_positions(best_positions, new_pop[cnt++]);
        }

        for (int i=0;i<cfg.POP_SIZE;++i) copy_positions(new_pop[i], population[i]);
        
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nMelhor global: %f\n", best_fit_global);
    printf("Tempo: %.3f s\n", elapsed_time);

    // Limpeza final
    liberar_populacao(population, cfg.POP_SIZE, cfg.N);
    liberar_populacao(new_pop, cfg.POP_SIZE, cfg.N);
    liberar_matrizes_avaliacao(evaluation_matrices, cfg.EVAL_MATRICES, cfg.N);
    liberar_matriz_int(best_positions, cfg.N);
    liberar_matriz_int(child_temp, cfg.N);
    liberar_matriz_int(mutated_temp, cfg.N);
    
    // Libera buffers otimizados
    liberar_matriz_double(buffer_A, cfg.N);
    liberar_matriz_double(buffer_aug, cfg.N);
    free(buffer_sol);

    free(min_connections_per_node);
    free(fitnesses);
    liberar_memoria_global();

    return 0;
}