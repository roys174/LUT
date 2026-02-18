/* SNN transformer architecture in pure C. */
/* Based on Spiking Manifesto (Izhikevich 2025) */
/* The code strives for simplicity, not efficiency. */
/* Eugene Izhikevich, October 2025 */
/* BPE support added February 2026 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define sign(x) (x > 0 ? 1 : -1)        // zero has "minus" sign
#define U(x) ( 0.5/(1+fabs(x)) )        // not used; here for reference
#define Up(x) (-0.5*sign(x)/(1+fabs(x))/(1+fabs(x)))

#define MAGIC_TOKENS 0x4C555442  // "LUTB"
#define MAGIC_VOCAB  0x564F4342  // "VOCB"

float learning_rate;

// ----------------------------------------------------------------------------
// Runtime configuration (replaces all #define constants)

typedef struct {
    int context_size;         // was CONTEXT_SIZE (32)
    int vocab_size;           // was VOCAB_SIZE (256), now from data file
    int embedding_dim;        // was EMBEDDING_DIM (32)
    int positional_dim;       // was POSITIONAL_DIM (4)
    int num_layers;           // was NUM_LAYERS (6)
    int num_heads;            // was NUM_HEADS (4)
    int n_t;                  // was N_T (16)
    int n_c;                  // was N_C (6)
    int testing_length;       // was TESTING_LENGTH (10000)
    int max_steps;            // was 100000000
    int validation_interval;  // was 10000
    float temperature;        // was 0.4
    int factored_output;      // 0 or 1
    int vocab_hi;             // (vocab_size + 255) / 256
    int shift_qk;             // n_c + positional_dim
    char* loss_file;          // was FILE_NAME "loss.csv"
} Config;

void init_config(Config* cfg) {
    cfg->context_size = 32;
    cfg->vocab_size = 0;       // set from data file
    cfg->embedding_dim = 32;
    cfg->positional_dim = 4;
    cfg->num_layers = 6;
    cfg->num_heads = 4;
    cfg->n_t = 16;
    cfg->n_c = 6;
    cfg->testing_length = 10000;
    cfg->max_steps = 100000000;
    cfg->validation_interval = 10000;
    cfg->temperature = 0.4f;
    cfg->factored_output = 0;
    cfg->vocab_hi = 0;
    cfg->shift_qk = 0;
    cfg->loss_file = "loss.csv";
}

void finalize_config(Config* cfg) {
    cfg->vocab_hi = (cfg->vocab_size + 255) / 256;
    cfg->shift_qk = cfg->n_c + cfg->positional_dim;
}

// ----------------------------------------------------------------------------
// Vocab for text output

typedef struct {
    char** strings;   // array of vocab_size string pointers
    int* lengths;     // length of each string
    char* pool;       // single allocation for all string data
    int vocab_size;
} Vocab;

void load_vocab(Vocab* v, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Error opening vocab file %s\n", path);
        exit(1);
    }

    unsigned int magic, vocab_size;
    fread(&magic, 4, 1, f);
    fread(&vocab_size, 4, 1, f);
    if (magic != MAGIC_VOCAB) {
        printf("Bad vocab magic: %x (expected %x)\n", magic, MAGIC_VOCAB);
        exit(1);
    }

    v->vocab_size = vocab_size;
    v->strings = (char**)malloc(vocab_size * sizeof(char*));
    v->lengths = (int*)malloc(vocab_size * sizeof(int));

    // First pass: compute total pool size
    long start = ftell(f);
    int total = 0;
    for (unsigned int i = 0; i < vocab_size; i++) {
        unsigned short len;
        fread(&len, 2, 1, f);
        total += len;
        fseek(f, len, SEEK_CUR);
    }

    // Allocate pool and second pass
    v->pool = (char*)malloc(total);
    fseek(f, start, SEEK_SET);
    char* ptr = v->pool;
    for (unsigned int i = 0; i < vocab_size; i++) {
        unsigned short len;
        fread(&len, 2, 1, f);
        v->strings[i] = ptr;
        v->lengths[i] = len;
        fread(ptr, 1, len, f);
        ptr += len;
    }
    fclose(f);
    printf("Loaded vocab: %d tokens\n", vocab_size);
}

void free_vocab(Vocab* v) {
    if (v->strings) free(v->strings);
    if (v->lengths) free(v->lengths);
    if (v->pool) free(v->pool);
    v->strings = NULL;
    v->lengths = NULL;
    v->pool = NULL;
}

// ----------------------------------------------------------------------------
// Training data

typedef struct {
    int* data;              // int32 token IDs
    int length;             // number of usable starting positions
    int total_tokens;       // total tokens loaded
    int vocab_size;         // from file header

    int* val_data;          // validation data (separate file or NULL)
    int val_length;
    int val_total_tokens;

    int* testing_input_data; // heap: testing_length indices into val_data (or data)
    unsigned char* reserved_for_testing; // only used when val_data is NULL
} TrainingData;

void load_bin_file(const char* path, int** out_data, int* out_total, int* out_vocab_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Error opening data file %s\n", path);
        exit(1);
    }
    unsigned int magic, vocab_size, num_tokens;
    fread(&magic, 4, 1, f);
    fread(&vocab_size, 4, 1, f);
    fread(&num_tokens, 4, 1, f);
    if (magic != MAGIC_TOKENS) {
        printf("Bad data magic: %x (expected %x)\n", magic, MAGIC_TOKENS);
        exit(1);
    }
    *out_data = (int*)malloc(num_tokens * sizeof(int));
    fread(*out_data, sizeof(int), num_tokens, f);
    fclose(f);
    *out_total = num_tokens;
    *out_vocab_size = vocab_size;
    printf("Loaded %s: %d tokens, vocab_size=%d\n", path, num_tokens, vocab_size);
}

void load_training_data(TrainingData* training, Config* cfg, const char* train_path, const char* val_path) {
    load_bin_file(train_path, &training->data, &training->total_tokens, &training->vocab_size);
    training->length = training->total_tokens - cfg->context_size - 1;

    training->val_data = NULL;
    training->val_length = 0;
    training->val_total_tokens = 0;
    training->reserved_for_testing = NULL;

    if (val_path) {
        int val_vocab_size;
        load_bin_file(val_path, &training->val_data, &training->val_total_tokens, &val_vocab_size);
        if (val_vocab_size != training->vocab_size) {
            printf("Warning: val vocab_size %d != train vocab_size %d\n", val_vocab_size, training->vocab_size);
        }
        training->val_length = training->val_total_tokens - cfg->context_size - 1;
    }

    // Set up testing indices
    training->testing_input_data = (int*)malloc(cfg->testing_length * sizeof(int));

    if (training->val_data) {
        // Sample testing positions from validation data
        for (int i = 0; i < cfg->testing_length; i++) {
            training->testing_input_data[i] = rand() % training->val_length;
        }
    } else {
        // Reserve testing positions from training data (original behavior)
        training->reserved_for_testing = (unsigned char*)calloc(training->length, sizeof(unsigned char));
        for (int i = 0; i < cfg->testing_length; i++) {
            training->testing_input_data[i] = rand() % training->length;
            for (int j = -cfg->context_size; j <= cfg->context_size; j++) {
                int idx = training->testing_input_data[i] + j;
                if (idx >= 0 && idx < training->length)
                    training->reserved_for_testing[idx] = 1;
            }
        }
    }
}

void free_training_data(TrainingData* training) {
    if (training->data) free(training->data);
    if (training->val_data) free(training->val_data);
    if (training->testing_input_data) free(training->testing_input_data);
    if (training->reserved_for_testing) free(training->reserved_for_testing);
}

int get_random_training_index(TrainingData* training) {
    int idx;
    if (training->reserved_for_testing) {
        do {
            idx = rand() % training->length;
        } while (training->reserved_for_testing[idx] == 1);
    } else {
        idx = rand() % training->length;
    }
    return idx;
}

// ----------------------------------------------------------------------------
// Spiking model

typedef struct {
    int* a;  // [n_c]
    int* b;  // [n_c]
} Anchors;

typedef struct {
    int y_dim;
    float** S;        // [n_t] pointers, each to (table_size * y_dim) floats
    Anchors* anchors; // [n_t]
    int n_t;
    int n_c;          // number of anchor comparisons (always cfg->n_c)
    int total_n_c;    // total bits for table indexing (may be larger for concatenated LUTs)
} LUT;

typedef struct {
    int* r_min;    // [n_t]
    float* u_min;  // [n_t]
    int* j;        // [n_t]
} LUTcache;

typedef struct {
    LUT V;
    LUTcache* V_cache;   // [context_size]
    float* PE;            // [context_size * n_t * positional_dim]
    LUTcache* PE_cache;   // [context_size]
} AttentionHead;

typedef struct {
    float* Token_embedder;  // [vocab_size * embedding_dim]
    int* tokens;            // [context_size + 1]
    float* z;               // [context_size * embedding_dim]

    LUT* FFN;               // [num_layers]
    LUTcache** FFN_cache;   // [num_layers][context_size]

    AttentionHead** head;   // [num_layers][num_heads]

    // Standard output
    LUT unembedder;
    LUTcache* unemb_cache; // [context_size]
    float* output;          // [context_size * vocab_size]

    // Factored output
    LUT unembedder_hi, unembedder_lo;
    LUTcache *unemb_hi_cache, *unemb_lo_cache;  // [context_size] each
    float *output_hi, *output_lo;                 // [context_size * vocab_hi] and [context_size * 256]
    int *tokens_hi, *tokens_lo;                   // [context_size + 1] each
} Model;


// ----------------------------------------------------------------------------
// Utility functions

void softmax(float* x, int size, float temperature) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf((x[i] - max_val) / temperature);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float vector_multiply(float* vector1, float* vector2, int size) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        result += vector1[i] * vector2[i];
    }
    return result;
}

void random_vector(float* vector, int size, float scale) {
    for (int i = 0; i < size; i++) {
        vector[i] = scale * 2 * ((float)rand()/RAND_MAX - 0.5f);
    }
}

int sample(float* probabilities, int n) {
    float coin = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

void fill_vector_with_random_intergers(int* vector, int N, int Max_value) {
    for (int i = 0; i < N; i++) {
        vector[i] = rand() % Max_value;
    }
}

void fill_vector_with_random_intergers_different_from_vector2(int* vector, int* vector2, int N, int Max_value) {
    for (int i = 0; i < N; i++) {
        do {
            vector[i] = rand() % Max_value;
        } while (vector2[i] == vector[i]);
    }
}

// ----------------------------------------------------------------------------
// LUT cache allocation

LUTcache alloc_LUTcache(int n_t) {
    LUTcache c;
    c.r_min = (int*)malloc(n_t * sizeof(int));
    c.u_min = (float*)malloc(n_t * sizeof(float));
    c.j = (int*)malloc(n_t * sizeof(int));
    return c;
}

void free_LUTcache(LUTcache* c) {
    free(c->r_min);
    free(c->u_min);
    free(c->j);
}

// ----------------------------------------------------------------------------
// LUT operations

void build_LUT(LUT* lut, Config* cfg, int total_n_c, int y_dim) {
    lut->y_dim = y_dim;
    lut->n_t = cfg->n_t;
    lut->n_c = cfg->n_c;       // anchor comparisons always use cfg->n_c
    lut->total_n_c = total_n_c; // total bits for table size
    lut->S = (float**)malloc(cfg->n_t * sizeof(float*));
    lut->anchors = (Anchors*)malloc(cfg->n_t * sizeof(Anchors));

    for (int i = 0; i < cfg->n_t; i++) {
        lut->anchors[i].a = (int*)malloc(cfg->n_c * sizeof(int));
        lut->anchors[i].b = (int*)malloc(cfg->n_c * sizeof(int));
        fill_vector_with_random_intergers(lut->anchors[i].a, cfg->n_c, cfg->embedding_dim);
        fill_vector_with_random_intergers_different_from_vector2(lut->anchors[i].b, lut->anchors[i].a, cfg->n_c, cfg->embedding_dim);
        lut->S[i] = (float*)calloc((1 << total_n_c) * y_dim, sizeof(float));
    }
}

void free_LUT(LUT* lut) {
    for (int i = 0; i < lut->n_t; i++) {
        free(lut->anchors[i].a);
        free(lut->anchors[i].b);
        free(lut->S[i]);
    }
    free(lut->S);
    free(lut->anchors);
}

void cache_index(LUT* lut, LUTcache* cache, float* x) {
    int n_t = lut->n_t;
    int n_c = lut->n_c;
    for (int i = 0; i < n_t; i++) {
        cache->j[i] = 0;
        cache->u_min[i] = INFINITY;
        for (int r = 0; r < n_c; r++) {
            float u = x[lut->anchors[i].a[r]] - x[lut->anchors[i].b[r]];
            if (u > 0) {
                cache->j[i] |= (1 << r);
            }
            if (fabs(u) < fabs(cache->u_min[i])) {
                cache->r_min[i] = r;
                cache->u_min[i] = u;
            }
        }
    }
}

void cache_PE_index(LUTcache* cache, float* pe, int n_t, int positional_dim) {
    // pe points to [n_t][positional_dim]
    for (int i = 0; i < n_t; i++) {
        cache->j[i] = 0;
        cache->u_min[i] = INFINITY;
        for (int r = 0; r < positional_dim; r++) {
            float u = pe[i * positional_dim + r];
            if (u > 0) {
                cache->j[i] |= (1 << r);
            }
            if (fabs(u) < fabs(cache->u_min[i])) {
                cache->r_min[i] = r;
                cache->u_min[i] = u;
            }
        }
    }
}

void LUT_forward(LUT* lut, LUTcache* cache, float* y) {
    for (int i = 0; i < lut->n_t; i++) {
        for (int k = 0; k < lut->y_dim; k++) {
            y[k] += lut->S[i][cache->j[i] * lut->y_dim + k];
        }
    }
}

#define BACKWARD_UPDATE(cache, gradient) \
    do { \
        float gi = 0; \
        for (int k = 0; k < lut->y_dim; k++) { \
            gi += y_gradient[k] * ( lut->S[i][ jbar + k ] - lut->S[i][ j + k ] ); \
        } \
        float v = gi * Up(cache->u_min[i]); \
        gradient[lut->anchors[i].a[cache->r_min[i]]] += v; \
        gradient[lut->anchors[i].b[cache->r_min[i]]] -= v; \
    } while (0)

void LUT_backward(LUT* lut, LUTcache* cache, float* x_gradient, float* y_gradient) {
    for (int i = 0; i < lut->n_t; i++) {
        int j = cache->j[i] * lut->y_dim;
        int jbar = (cache->j[i] ^ (1 << cache->r_min[i])) * lut->y_dim;
        BACKWARD_UPDATE(cache, x_gradient);
        for (int k = 0; k < lut->y_dim; k++) {
            lut->S[i][j + k] -= learning_rate * y_gradient[k];
        }
    }
}

// Concatenated index: ((Q << shift_qk) | (P << positional_dim) | PE) * y_dim
static inline int concatenate_idx(int Q, int P, int PE, int shift_qk, int positional_dim, int y_dim) {
    return (((Q) << shift_qk) | ((P) << positional_dim) | (PE)) * y_dim;
}

void concatenated_LUT_forward(LUT* lut, Config* cfg, LUTcache* cacheQ, LUTcache* cacheK, LUTcache* cachePE, float* y) {
    int shift_qk = cfg->shift_qk;
    int pdim = cfg->positional_dim;
    for (int i = 0; i < lut->n_t; i++) {
        int j = concatenate_idx(cacheQ->j[i], cacheK->j[i], cachePE->j[i], shift_qk, pdim, lut->y_dim);
        for (int k = 0; k < lut->y_dim; k++) {
            y[k] += lut->S[i][j + k];
        }
    }
}

void concatenated_LUT_backward(LUT* lut, Config* cfg, LUTcache* cacheQ, LUTcache* cacheK, LUTcache* cachePE,
                                float* x_gradientQ, float* x_gradientK, float* PE_grad, float* y_gradient) {
    // PE_grad is [n_t * positional_dim]
    int shift_qk = cfg->shift_qk;
    int pdim = cfg->positional_dim;
    int yd = lut->y_dim;

    for (int i = 0; i < lut->n_t; i++) {
        int j = concatenate_idx(cacheQ->j[i], cacheK->j[i], cachePE->j[i], shift_qk, pdim, yd);

        if (fabs(cacheQ->u_min[i]) < fabs(cacheK->u_min[i])) {
            int jbar = concatenate_idx(cacheQ->j[i] ^ (1 << cacheQ->r_min[i]), cacheK->j[i], cachePE->j[i], shift_qk, pdim, yd);
            BACKWARD_UPDATE(cacheQ, x_gradientQ);
        } else {
            int jbar = concatenate_idx(cacheQ->j[i], cacheK->j[i] ^ (1 << cacheK->r_min[i]), cachePE->j[i], shift_qk, pdim, yd);
            BACKWARD_UPDATE(cacheK, x_gradientK);
        }

        if (fabs(cachePE->u_min[i]) < fabs(cacheQ->u_min[i]) && fabs(cachePE->u_min[i]) < fabs(cacheK->u_min[i])) {
            int jbarPE = concatenate_idx(cacheQ->j[i], cacheK->j[i], cachePE->j[i] ^ (1 << cachePE->r_min[i]), shift_qk, pdim, yd);
            float giPE = 0;
            for (int k = 0; k < yd; k++) {
                giPE += y_gradient[k] * (lut->S[i][jbarPE + k] - lut->S[i][j + k]);
            }
            float deltaPE = giPE * Up(cachePE->u_min[i]);
            PE_grad[i * pdim + cachePE->r_min[i]] += deltaPE;
        }

        for (int k = 0; k < yd; k++) {
            lut->S[i][j + k] -= learning_rate * y_gradient[k];
        }
    }
}

// ----------------------------------------------------------------------------
// Model build / free

void build_Model(Model* m, Config* cfg) {
    int cs = cfg->context_size;
    int ed = cfg->embedding_dim;
    int vs = cfg->vocab_size;
    int n_t = cfg->n_t;
    int n_c = cfg->n_c;
    int pdim = cfg->positional_dim;

    // Token embedder
    m->Token_embedder = (float*)malloc(vs * ed * sizeof(float));
    random_vector(m->Token_embedder, vs * ed, 1.0f);

    // Tokens buffer
    m->tokens = (int*)malloc((cs + 1) * sizeof(int));

    // Residual stream
    m->z = (float*)calloc(cs * ed, sizeof(float));

    // FFN layers
    m->FFN = (LUT*)malloc(cfg->num_layers * sizeof(LUT));
    m->FFN_cache = (LUTcache**)malloc(cfg->num_layers * sizeof(LUTcache*));
    for (int l = 0; l < cfg->num_layers; l++) {
        build_LUT(&m->FFN[l], cfg, n_c, ed);
        m->FFN_cache[l] = (LUTcache*)malloc(cs * sizeof(LUTcache));
        for (int pos = 0; pos < cs; pos++) {
            m->FFN_cache[l][pos] = alloc_LUTcache(n_t);
        }
    }

    // Attention heads
    int total_attn_nc = n_c + n_c + pdim;
    m->head = (AttentionHead**)malloc(cfg->num_layers * sizeof(AttentionHead*));
    for (int l = 0; l < cfg->num_layers; l++) {
        m->head[l] = (AttentionHead*)malloc(cfg->num_heads * sizeof(AttentionHead));
        for (int h = 0; h < cfg->num_heads; h++) {
            AttentionHead* ah = &m->head[l][h];
            // PE: [context_size][n_t][positional_dim]
            ah->PE = (float*)malloc(cs * n_t * pdim * sizeof(float));
            random_vector(ah->PE, cs * n_t * pdim, 1.0f);
            // V LUT
            build_LUT(&ah->V, cfg, total_attn_nc, ed);
            // Caches
            ah->V_cache = (LUTcache*)malloc(cs * sizeof(LUTcache));
            ah->PE_cache = (LUTcache*)malloc(cs * sizeof(LUTcache));
            for (int pos = 0; pos < cs; pos++) {
                ah->V_cache[pos] = alloc_LUTcache(n_t);
                ah->PE_cache[pos] = alloc_LUTcache(n_t);
            }
        }
    }

    // Output head
    if (cfg->factored_output) {
        int vh = cfg->vocab_hi;
        build_LUT(&m->unembedder_hi, cfg, n_c, vh);
        build_LUT(&m->unembedder_lo, cfg, n_c, 256);
        m->unemb_hi_cache = (LUTcache*)malloc(cs * sizeof(LUTcache));
        m->unemb_lo_cache = (LUTcache*)malloc(cs * sizeof(LUTcache));
        for (int pos = 0; pos < cs; pos++) {
            m->unemb_hi_cache[pos] = alloc_LUTcache(n_t);
            m->unemb_lo_cache[pos] = alloc_LUTcache(n_t);
        }
        m->output_hi = (float*)calloc(cs * vh, sizeof(float));
        m->output_lo = (float*)calloc(cs * 256, sizeof(float));
        m->tokens_hi = (int*)malloc((cs + 1) * sizeof(int));
        m->tokens_lo = (int*)malloc((cs + 1) * sizeof(int));
        // NULL out standard fields
        m->unemb_cache = NULL;
        m->output = NULL;
    } else {
        build_LUT(&m->unembedder, cfg, n_c, vs);
        m->unemb_cache = (LUTcache*)malloc(cs * sizeof(LUTcache));
        for (int pos = 0; pos < cs; pos++) {
            m->unemb_cache[pos] = alloc_LUTcache(n_t);
        }
        m->output = (float*)calloc(cs * vs, sizeof(float));
        // NULL out factored fields
        m->unemb_hi_cache = NULL;
        m->unemb_lo_cache = NULL;
        m->output_hi = NULL;
        m->output_lo = NULL;
        m->tokens_hi = NULL;
        m->tokens_lo = NULL;
    }
}

void free_Model(Model* m, Config* cfg) {
    int cs = cfg->context_size;

    free(m->Token_embedder);
    free(m->tokens);
    free(m->z);

    for (int l = 0; l < cfg->num_layers; l++) {
        free_LUT(&m->FFN[l]);
        for (int pos = 0; pos < cs; pos++) {
            free_LUTcache(&m->FFN_cache[l][pos]);
        }
        free(m->FFN_cache[l]);
    }
    free(m->FFN);
    free(m->FFN_cache);

    for (int l = 0; l < cfg->num_layers; l++) {
        for (int h = 0; h < cfg->num_heads; h++) {
            AttentionHead* ah = &m->head[l][h];
            free(ah->PE);
            free_LUT(&ah->V);
            for (int pos = 0; pos < cs; pos++) {
                free_LUTcache(&ah->V_cache[pos]);
                free_LUTcache(&ah->PE_cache[pos]);
            }
            free(ah->V_cache);
            free(ah->PE_cache);
        }
        free(m->head[l]);
    }
    free(m->head);

    if (cfg->factored_output) {
        free_LUT(&m->unembedder_hi);
        free_LUT(&m->unembedder_lo);
        for (int pos = 0; pos < cs; pos++) {
            free_LUTcache(&m->unemb_hi_cache[pos]);
            free_LUTcache(&m->unemb_lo_cache[pos]);
        }
        free(m->unemb_hi_cache);
        free(m->unemb_lo_cache);
        free(m->output_hi);
        free(m->output_lo);
        free(m->tokens_hi);
        free(m->tokens_lo);
    } else {
        free_LUT(&m->unembedder);
        for (int pos = 0; pos < cs; pos++) {
            free_LUTcache(&m->unemb_cache[pos]);
        }
        free(m->unemb_cache);
        free(m->output);
    }
}

// ----------------------------------------------------------------------------
// Forward / backward

void attention_forward(AttentionHead* head, Config* cfg, float* x, float* y) {
    // x, y are [context_size][embedding_dim]
    int cs = cfg->context_size;
    int ed = cfg->embedding_dim;
    int n_t = cfg->n_t;
    int pdim = cfg->positional_dim;

    for (int pos = 0; pos < cs; pos++) {
        cache_index(&head->V, &head->V_cache[pos], x + pos * ed);
        cache_PE_index(&head->PE_cache[pos], head->PE + pos * n_t * pdim, n_t, pdim);
    }

    for (int pos = 1; pos < cs; pos++) {
        for (int pos1 = 0; pos1 < pos; pos1++) {
            concatenated_LUT_forward(&head->V, cfg,
                &head->V_cache[pos], &head->V_cache[pos1], &head->PE_cache[pos - pos1],
                y + pos * ed);
        }
    }
}

void attention_backward(AttentionHead* head, Config* cfg, float* x_grad, float* y_grad) {
    int cs = cfg->context_size;
    int ed = cfg->embedding_dim;
    int n_t = cfg->n_t;
    int pdim = cfg->positional_dim;

    // pos_grad: [context_size][n_t][positional_dim]
    float* pos_grad = (float*)calloc(cs * n_t * pdim, sizeof(float));

    for (int pos = 1; pos < cs; pos++) {
        for (int pos1 = 0; pos1 < pos; pos1++) {
            concatenated_LUT_backward(&head->V, cfg,
                &head->V_cache[pos], &head->V_cache[pos1], &head->PE_cache[pos - pos1],
                x_grad + pos * ed, x_grad + pos1 * ed,
                pos_grad + (pos - pos1) * n_t * pdim,
                y_grad + pos * ed);
        }
    }

    // Update PE
    for (int pos = 0; pos < cs; pos++) {
        for (int i = 0; i < n_t; i++) {
            for (int k = 0; k < pdim; k++) {
                head->PE[pos * n_t * pdim + i * pdim + k] -= learning_rate * pos_grad[pos * n_t * pdim + i * pdim + k];
            }
        }
    }
    free(pos_grad);
}

void embed_token(Model* m, Config* cfg, int token_id, float* output) {
    memcpy(output, m->Token_embedder + token_id * cfg->embedding_dim, cfg->embedding_dim * sizeof(float));
}

void load_snippet(Model* m, Config* cfg, int* data, int start) {
    int cs = cfg->context_size;
    int ed = cfg->embedding_dim;
    for (int pos = 0; pos < cs; pos++) {
        embed_token(m, cfg, data[start + pos], m->z + pos * ed);
        m->tokens[pos] = data[start + pos];
    }
    m->tokens[cs] = data[start + cs];

    if (cfg->factored_output) {
        for (int pos = 0; pos <= cs; pos++) {
            m->tokens_hi[pos] = m->tokens[pos] / 256;
            m->tokens_lo[pos] = m->tokens[pos] % 256;
        }
    }
}

void model_forward(Model* m, Config* cfg) {
    int cs = cfg->context_size;
    int ed = cfg->embedding_dim;

    for (int l = 0; l < cfg->num_layers; l++) {
        // Attention: from all z to all z
        float* x = (float*)malloc(cs * ed * sizeof(float));
        memcpy(x, m->z, cs * ed * sizeof(float));
        for (int h = 0; h < cfg->num_heads; h++) {
            attention_forward(&m->head[l][h], cfg, x, m->z);
        }
        free(x);

        // FFN from z_pos to z_pos
        for (int pos = 0; pos < cs; pos++) {
            cache_index(&m->FFN[l], &m->FFN_cache[l][pos], m->z + pos * ed);
            LUT_forward(&m->FFN[l], &m->FFN_cache[l][pos], m->z + pos * ed);
        }
    }

    // Output head
    if (cfg->factored_output) {
        int vh = cfg->vocab_hi;
        memset(m->output_hi, 0, cs * vh * sizeof(float));
        memset(m->output_lo, 0, cs * 256 * sizeof(float));
        for (int pos = 0; pos < cs; pos++) {
            cache_index(&m->unembedder_hi, &m->unemb_hi_cache[pos], m->z + pos * ed);
            LUT_forward(&m->unembedder_hi, &m->unemb_hi_cache[pos], m->output_hi + pos * vh);
            cache_index(&m->unembedder_lo, &m->unemb_lo_cache[pos], m->z + pos * ed);
            LUT_forward(&m->unembedder_lo, &m->unemb_lo_cache[pos], m->output_lo + pos * 256);
        }
    } else {
        int vs = cfg->vocab_size;
        memset(m->output, 0, cs * vs * sizeof(float));
        for (int pos = 0; pos < cs; pos++) {
            cache_index(&m->unembedder, &m->unemb_cache[pos], m->z + pos * ed);
            LUT_forward(&m->unembedder, &m->unemb_cache[pos], m->output + pos * vs);
        }
    }
}

void model_backward(Model* m, Config* cfg) {
    int cs = cfg->context_size;
    int ed = cfg->embedding_dim;

    float* y_grad = (float*)malloc(cs * ed * sizeof(float));
    float* x_grad = (float*)calloc(cs * ed, sizeof(float));

    // Output head backward
    if (cfg->factored_output) {
        for (int pos = 0; pos < cs; pos++) {
            LUT_backward(&m->unembedder_hi, &m->unemb_hi_cache[pos], x_grad + pos * ed, m->output_hi + pos * cfg->vocab_hi);
            LUT_backward(&m->unembedder_lo, &m->unemb_lo_cache[pos], x_grad + pos * ed, m->output_lo + pos * 256);
        }
    } else {
        for (int pos = 0; pos < cs; pos++) {
            LUT_backward(&m->unembedder, &m->unemb_cache[pos], x_grad + pos * ed, m->output + pos * cfg->vocab_size);
        }
    }

    for (int l = cfg->num_layers - 1; l >= 0; l--) {
        // FFN
        memcpy(y_grad, x_grad, cs * ed * sizeof(float));
        for (int pos = 0; pos < cs; pos++) {
            LUT_backward(&m->FFN[l], &m->FFN_cache[l][pos], x_grad + pos * ed, y_grad + pos * ed);
        }

        // Attention
        memcpy(y_grad, x_grad, cs * ed * sizeof(float));
        for (int h = 0; h < cfg->num_heads; h++) {
            attention_backward(&m->head[l][h], cfg, x_grad, y_grad);
        }
    }

    // Token embedder gradient (disabled in original, kept disabled)
    // for (int pos = 0; pos < cs; pos++) {
    //     for (int k = 0; k < ed; k++) {
    //         m->Token_embedder[m->tokens[pos] * ed + k] -= learning_rate * x_grad[pos * ed + k];
    //     }
    // }

    free(y_grad);
    free(x_grad);
}

void model_compute_gradients(Model* m, Config* cfg) {
    int cs = cfg->context_size;
    if (cfg->factored_output) {
        int vh = cfg->vocab_hi;
        for (int pos = 0; pos < cs; pos++) {
            softmax(m->output_hi + pos * vh, vh, 1.0f);
            m->output_hi[pos * vh + m->tokens_hi[pos + 1]] -= 1.0f;
            softmax(m->output_lo + pos * 256, 256, 1.0f);
            m->output_lo[pos * 256 + m->tokens_lo[pos + 1]] -= 1.0f;
        }
    } else {
        int vs = cfg->vocab_size;
        for (int pos = 0; pos < cs; pos++) {
            softmax(m->output + pos * vs, vs, 1.0f);
            m->output[pos * vs + m->tokens[pos + 1]] -= 1.0f;
        }
    }
}

void model_training_step(Model* m, Config* cfg) {
    model_forward(m, cfg);
    model_compute_gradients(m, cfg);
    model_backward(m, cfg);
}

// ----------------------------------------------------------------------------
// Inference and sampling

int model_inference_standard(Model* m, Config* cfg) {
    model_forward(m, cfg);
    int vs = cfg->vocab_size;
    int last = (cfg->context_size - 1) * vs;
    softmax(m->output + last, vs, cfg->temperature);
    return sample(m->output + last, vs);
}

int model_inference_factored(Model* m, Config* cfg) {
    model_forward(m, cfg);
    int vh = cfg->vocab_hi;
    int last_hi = (cfg->context_size - 1) * vh;
    int last_lo = (cfg->context_size - 1) * 256;
    softmax(m->output_hi + last_hi, vh, cfg->temperature);
    softmax(m->output_lo + last_lo, 256, cfg->temperature);
    int hi = sample(m->output_hi + last_hi, vh);
    int lo = sample(m->output_lo + last_lo, 256);
    int token = hi * 256 + lo;
    if (token >= cfg->vocab_size) token = cfg->vocab_size - 1;
    return token;
}

int model_inference(Model* m, Config* cfg) {
    if (cfg->factored_output) {
        return model_inference_factored(m, cfg);
    } else {
        return model_inference_standard(m, cfg);
    }
}

void model_prompt_response(Model* m, Config* cfg, Vocab* vocab, int* prompt_tokens, int prompt_len, int response_length) {
    int cs = cfg->context_size;
    int ed = cfg->embedding_dim;

    // Build working buffer of context_size tokens
    int* buf = (int*)malloc(cs * sizeof(int));
    if (prompt_len >= cs) {
        memcpy(buf, prompt_tokens + prompt_len - cs, cs * sizeof(int));
    } else {
        // Pad with zeros (or first token)
        memset(buf, 0, cs * sizeof(int));
        memcpy(buf + cs - prompt_len, prompt_tokens, prompt_len * sizeof(int));
    }

    // Print the prompt
    if (vocab) {
        for (int i = 0; i < prompt_len; i++) {
            int tok = prompt_tokens[i];
            if (tok >= 0 && tok < vocab->vocab_size) {
                fwrite(vocab->strings[tok], 1, vocab->lengths[tok], stdout);
            }
        }
    }

    for (int i = 0; i < response_length; i++) {
        // Embed buffer into z
        for (int pos = 0; pos < cs; pos++) {
            embed_token(m, cfg, buf[pos], m->z + pos * ed);
        }
        int response = model_inference(m, cfg);

        if (vocab && response >= 0 && response < vocab->vocab_size) {
            fwrite(vocab->strings[response], 1, vocab->lengths[response], stdout);
        } else {
            printf("[%d]", response);
        }

        // Shift buffer
        for (int j = 0; j < cs - 1; j++) {
            buf[j] = buf[j + 1];
        }
        buf[cs - 1] = response;
    }
    free(buf);
}

// ----------------------------------------------------------------------------
// Validation loss

float compute_validation_loss(Model* m, Config* cfg, TrainingData* training) {
    int cs = cfg->context_size;
    float total_loss = 0;

    int* val_src = training->val_data ? training->val_data : training->data;

    for (int i = 0; i < cfg->testing_length; i++) {
        load_snippet(m, cfg, val_src, training->testing_input_data[i]);
        model_forward(m, cfg);

        if (cfg->factored_output) {
            int vh = cfg->vocab_hi;
            int last_hi = (cs - 1) * vh;
            int last_lo = (cs - 1) * 256;
            // Copy before softmax modifies
            softmax(m->output_hi + last_hi, vh, 1.0f);
            softmax(m->output_lo + last_lo, 256, 1.0f);
            float p_hi = m->output_hi[last_hi + m->tokens_hi[cs]];
            float p_lo = m->output_lo[last_lo + m->tokens_lo[cs]];
            total_loss += -logf(fmaxf(p_hi, 1e-10f)) + -logf(fmaxf(p_lo, 1e-10f));
        } else {
            int vs = cfg->vocab_size;
            int last = (cs - 1) * vs;
            softmax(m->output + last, vs, 1.0f);
            float p = m->output[last + m->tokens[cs]];
            total_loss += -logf(fmaxf(p, 1e-10f));
        }
    }
    return total_loss / cfg->testing_length;
}

// ----------------------------------------------------------------------------
// CLI parsing

void print_usage(const char* prog) {
    printf("Usage: %s train.bin [options]\n", prog);
    printf("  --val val.bin             validation data file\n");
    printf("  --vocab gpt2.vocab        vocabulary for text output\n");
    printf("  --context-size N          context window size (default: 32)\n");
    printf("  --embedding-dim N         embedding dimension (default: 32)\n");
    printf("  --positional-dim N        positional encoding dim (default: 4)\n");
    printf("  --num-layers N            number of layers (default: 6)\n");
    printf("  --num-heads N             number of attention heads (default: 4)\n");
    printf("  --n-t N                   number of tables per LUT (default: 16)\n");
    printf("  --n-c N                   number of comparisons per table (default: 6)\n");
    printf("  --testing-length N        validation samples (default: 10000)\n");
    printf("  --max-steps N             training steps (default: 100000000)\n");
    printf("  --validation-interval N   steps between validation (default: 10000)\n");
    printf("  --temperature F           sampling temperature (default: 0.4)\n");
    printf("  --factored-output         use factored base-256 output head\n");
    printf("  --loss-file FILE          loss output file (default: loss.csv)\n");
}

typedef struct {
    char* train_path;
    char* val_path;
    char* vocab_path;
} CLIPaths;

void parse_args(int argc, char** argv, Config* cfg, CLIPaths* paths) {
    paths->train_path = NULL;
    paths->val_path = NULL;
    paths->vocab_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (!paths->train_path) {
                paths->train_path = argv[i];
                continue;
            }
        }
        if (strcmp(argv[i], "--val") == 0 && i + 1 < argc) {
            paths->val_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            paths->vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--context-size") == 0 && i + 1 < argc) {
            cfg->context_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--embedding-dim") == 0 && i + 1 < argc) {
            cfg->embedding_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--positional-dim") == 0 && i + 1 < argc) {
            cfg->positional_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--num-layers") == 0 && i + 1 < argc) {
            cfg->num_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--num-heads") == 0 && i + 1 < argc) {
            cfg->num_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-t") == 0 && i + 1 < argc) {
            cfg->n_t = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-c") == 0 && i + 1 < argc) {
            cfg->n_c = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--testing-length") == 0 && i + 1 < argc) {
            cfg->testing_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) {
            cfg->max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--validation-interval") == 0 && i + 1 < argc) {
            cfg->validation_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            cfg->temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--factored-output") == 0) {
            cfg->factored_output = 1;
        } else if (strcmp(argv[i], "--loss-file") == 0 && i + 1 < argc) {
            cfg->loss_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    if (!paths->train_path) {
        print_usage(argv[0]);
        exit(1);
    }
}

// =================================================================================
// Main function
// =================================================================================
int main(int argc, char *argv[]) {

    Config cfg;
    CLIPaths paths;
    init_config(&cfg);
    parse_args(argc, argv, &cfg, &paths);

    // Load training data (sets vocab_size from file header)
    TrainingData training;
    load_training_data(&training, &cfg, paths.train_path, paths.val_path);
    cfg.vocab_size = training.vocab_size;
    finalize_config(&cfg);

    printf("Config: context_size=%d, vocab_size=%d, embedding_dim=%d, "
           "num_layers=%d, num_heads=%d, n_t=%d, n_c=%d, factored=%d\n",
           cfg.context_size, cfg.vocab_size, cfg.embedding_dim,
           cfg.num_layers, cfg.num_heads, cfg.n_t, cfg.n_c, cfg.factored_output);
    if (cfg.factored_output) {
        printf("Factored output: vocab_hi=%d, vocab_lo=256\n", cfg.vocab_hi);
    }

    // Load vocab if provided
    Vocab vocab = {0};
    int have_vocab = 0;
    if (paths.vocab_path) {
        load_vocab(&vocab, paths.vocab_path);
        have_vocab = 1;
    }

    // Initialize loss file
    FILE *file_loss = fopen(cfg.loss_file, "w"); fclose(file_loss);

    // Build model
    Model m;
    build_Model(&m, &cfg);

    // Training loop
    for (int t = 0; t < cfg.max_steps; t++) {

        load_snippet(&m, &cfg, training.data, get_random_training_index(&training));
        learning_rate = MIN(1.0f / sqrtf(1.0f + t), (float)t / 4000.0f / sqrtf(4000.0f));
        model_training_step(&m, &cfg);

        if (t % cfg.validation_interval == 0) {

            printf("...validating... "); fflush(stdout);
            float validation_loss = compute_validation_loss(&m, &cfg, &training);

            FILE *file_loss = fopen(cfg.loss_file, "a");
            fprintf(file_loss, "%d, %f\n", t, validation_loss);
            fclose(file_loss);

            printf("\rt=%d,000, loss=%5.3f: ", t / 1000, validation_loss);

            // Generate sample text
            if (have_vocab) {
                // Use a simple prompt from validation data
                int* val_src = training.val_data ? training.val_data : training.data;
                int prompt_start = training.testing_input_data[0];
                int prompt_len = MIN(cfg.context_size, 8);
                model_prompt_response(&m, &cfg, &vocab, val_src + prompt_start, prompt_len, 80);
            } else {
                printf("[no vocab loaded]");
            }
            printf("\n");
        }
        printf("\rt=%d", t); fflush(stdout);
    }

    free_Model(&m, &cfg);
    free_training_data(&training);
    if (have_vocab) free_vocab(&vocab);
    return 0;
}
