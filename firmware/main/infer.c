#include "infer.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"

static const char *TAG = "eslm";

typedef struct {
    int vocab, d, n_layer, n_head, ctx;
} Config;

typedef struct {
    const float *f;      // f32 weights (when s == NULL)
    const int8_t *q;     // int8 weights
    const float *s;      // per-output-channel scales (int8 only)
    int out, in;
} Mat;

struct eslm {
    Config c;
    int hd, mlp, dtype, version, n_params;

    Mat tok_emb, pos_emb;
    Mat *wq, *wk, *wv, *wo, *w1, *w2;
    const float **rms1, **rms2, *rms_f;

    const uint32_t *tok_offs;
    const uint8_t *tok_blob;
    int sp_u, sp_a, sp_e;

    float *k_cache, *v_cache;
    float *x, *xb, *q, *k, *v, *att, *hb, *logits, *scores, *probs;

    void *weights;              // PSRAM copy of the weights (NULL -> flash)
    uint32_t rng;
};

static void *alloc_psram(size_t n) {
    void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_malloc(n, MALLOC_CAP_8BIT);
    return p;
}

static void *alloc_internal(size_t n) {
    return heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

static float *f32(size_t n) { return (float *)alloc_internal(n * sizeof(float)); }

static void matvec(const Mat *m, const float *x, float *y) {
    const int out = m->out, in = m->in;
    if (!m->s) {
        const float *w = m->f;
        for (int i = 0; i < out; i++) {
            const float *row = w + (size_t)i * in;
            float acc = 0.0f;
            for (int j = 0; j < in; j++) acc += row[j] * x[j];
            y[i] = acc;
        }
    } else {
        const int8_t *w = m->q;
        for (int i = 0; i < out; i++) {
            const int8_t *row = w + (size_t)i * in;
            float acc = 0.0f;
            for (int j = 0; j < in; j++) acc += (float)row[j] * x[j];
            y[i] = acc * m->s[i];
        }
    }
}

static float row_dot(const Mat *m, int row, const float *x) {
    const int in = m->in;
    if (!m->s) {
        const float *r = m->f + (size_t)row * in;
        float acc = 0.0f;
        for (int j = 0; j < in; j++) acc += r[j] * x[j];
        return acc;
    }
    const int8_t *r = m->q + (size_t)row * in;
    float acc = 0.0f;
    for (int j = 0; j < in; j++) acc += (float)r[j] * x[j];
    return acc * m->s[row];
}

static void emb_add(const Mat *m, int row, float *dst) {
    const int in = m->in;
    if (!m->s) {
        const float *r = m->f + (size_t)row * in;
        for (int i = 0; i < in; i++) dst[i] += r[i];
    } else {
        const float sc = m->s[row];
        const int8_t *r = m->q + (size_t)row * in;
        for (int i = 0; i < in; i++) dst[i] += (float)r[i] * sc;
    }
}

static void rmsnorm(const float *v, const float *w, float *out, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = v[i] * inv * w[i];
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static const uint8_t *rd_mat(const uint8_t *cur, Mat *m, int out, int in, int dtype) {
    if (dtype == 1) {
        m->s = (const float *)cur;
        cur += (size_t)out * 4;
        m->q = (const int8_t *)cur;
        cur += (size_t)out * in;
        cur += (-(out * in)) & 3;
    } else {
        m->f = (const float *)cur;
        cur += (size_t)out * in * 4;
    }
    m->out = out;
    m->in = in;
    return cur;
}

static const uint8_t *rd_vec(const uint8_t *cur, const float **v, int n) {
    *v = (const float *)cur;
    return cur + (size_t)n * 4;
}

eslm_t *eslm_init(const uint8_t *blob, size_t len) {
    if (len < 48 || memcmp(blob, "ESLM", 4) != 0) {
        ESP_LOGE(TAG, "bad model magic");
        return NULL;
    }
    const uint32_t *h = (const uint32_t *)(blob + 4);
    uint32_t version = h[0], dtype = h[1];
    if (version != 1 && version != 2) {
        ESP_LOGE(TAG, "unsupported version %u", (unsigned)version);
        return NULL;
    }
    if (dtype > 1) {
        ESP_LOGE(TAG, "unsupported dtype %u", (unsigned)dtype);
        return NULL;
    }

    eslm_t *m = (eslm_t *)calloc(1, sizeof(eslm_t));
    if (!m) return NULL;
    m->version = (int)version;
    m->dtype = (int)dtype;
    m->c.vocab = (int)h[2];
    m->c.d = (int)h[3];
    m->c.n_layer = (int)h[4];
    m->c.n_head = (int)h[5];
    m->c.ctx = (int)h[6];
    m->hd = m->c.d / m->c.n_head;
    m->mlp = 4 * m->c.d;

    size_t hdr = (version == 1) ? 36 : 48;
    const uint8_t *payload = blob + hdr;
    if (version == 2) {
        m->sp_u = (int)h[7];
        m->sp_a = (int)h[8];
        m->sp_e = (int)h[9];
        m->tok_offs = (const uint32_t *)(blob + 48);
        m->tok_blob = blob + 48 + 4u * (uint32_t)(m->c.vocab + 1);
        payload = m->tok_blob + h[10];
        ESP_LOGI(TAG, "v2 blob: vocab %d, specials u=%d a=%d e=%d",
                 m->c.vocab, m->sp_u, m->sp_a, m->sp_e);
    }

    size_t wbytes = len - (size_t)(payload - blob);
    m->weights = alloc_psram(wbytes);
    if (m->weights) {
        memcpy(m->weights, payload, wbytes);
    } else {
        ESP_LOGW(TAG, "no PSRAM for weights, using flash XIP");
    }
    const uint8_t *cur = m->weights ? (const uint8_t *)m->weights : payload;

    const int L = m->c.n_layer, d = m->c.d, V = m->c.vocab, ctx = m->c.ctx, mlp = m->mlp;
    cur = rd_mat(cur, &m->tok_emb, V, d, m->dtype);
    cur = rd_mat(cur, &m->pos_emb, ctx, d, m->dtype);

    m->rms1 = calloc(L, sizeof(float *));
    m->rms2 = calloc(L, sizeof(float *));
    m->wq = calloc(L, sizeof(Mat));
    m->wk = calloc(L, sizeof(Mat));
    m->wv = calloc(L, sizeof(Mat));
    m->wo = calloc(L, sizeof(Mat));
    m->w1 = calloc(L, sizeof(Mat));
    m->w2 = calloc(L, sizeof(Mat));
    for (int l = 0; l < L; l++) {
        cur = rd_vec(cur, &m->rms1[l], d);
        cur = rd_mat(cur, &m->wq[l], d, d, m->dtype);
        cur = rd_mat(cur, &m->wk[l], d, d, m->dtype);
        cur = rd_mat(cur, &m->wv[l], d, d, m->dtype);
        cur = rd_mat(cur, &m->wo[l], d, d, m->dtype);
        cur = rd_vec(cur, &m->rms2[l], d);
        cur = rd_mat(cur, &m->w1[l], mlp, d, m->dtype);
        cur = rd_mat(cur, &m->w2[l], d, mlp, m->dtype);
    }
    cur = rd_vec(cur, &m->rms_f, d);

    m->n_params = V * d + ctx * d + L * (4 * d * d + 2 * d + 8 * d * d) + d;

    m->k_cache = (float *)alloc_psram((size_t)L * ctx * d * sizeof(float));
    m->v_cache = (float *)alloc_psram((size_t)L * ctx * d * sizeof(float));
    m->x = f32(d);
    m->xb = f32(mlp > d ? mlp : d);
    m->q = f32(d);
    m->k = f32(d);
    m->v = f32(d);
    m->att = f32(d);
    m->hb = f32(mlp);
    m->logits = f32(V);
    m->scores = f32(ctx);
    m->probs = f32(V);
    m->rng = 1234567u;

    if (!m->k_cache || !m->v_cache || !m->x || !m->logits) {
        ESP_LOGE(TAG, "out of memory (free internal=%u psram=%u)",
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
        return NULL;
    }
    ESP_LOGI(TAG, "model %d params, %d layers, ctx %d, %s weights, %s",
             m->n_params, L, ctx, m->dtype ? "int8" : "f32",
             m->weights ? "PSRAM copy" : "flash XIP");
    return m;
}

void eslm_deinit(eslm_t *m) {
    if (!m) return;
    free(m->weights);
    free(m->k_cache);
    free(m->v_cache);
    free((void *)m->rms1); free((void *)m->rms2);
    free(m->wq); free(m->wk); free(m->wv); free(m->wo); free(m->w1); free(m->w2);
    free(m->x); free(m->xb); free(m->q); free(m->k); free(m->v); free(m->att);
    free(m->hb); free(m->logits); free(m->scores); free(m->probs);
    free(m);
}

int eslm_ctx(const eslm_t *m) { return m->c.ctx; }
int eslm_vocab(const eslm_t *m) { return m->c.vocab; }
int eslm_params(const eslm_t *m) { return m->n_params; }
int eslm_dtype(const eslm_t *m) { return m->dtype; }
void eslm_set_seed(eslm_t *m, uint32_t seed) { m->rng = seed ? seed : 1u; }

void eslm_reset(eslm_t *m) {
    memset(m->k_cache, 0, (size_t)m->c.n_layer * m->c.ctx * m->c.d * sizeof(float));
    memset(m->v_cache, 0, (size_t)m->c.n_layer * m->c.ctx * m->c.d * sizeof(float));
}

float eslm_forward(eslm_t *m, int token, int pos) {
    const int d = m->c.d, L = m->c.n_layer, nh = m->c.n_head, hd = m->hd;
    const int mlp = m->mlp, ctx = m->c.ctx;

    for (int i = 0; i < d; i++) m->x[i] = 0.0f;
    emb_add(&m->tok_emb, token, m->x);
    emb_add(&m->pos_emb, pos % ctx, m->x);

    for (int l = 0; l < L; l++) {
        rmsnorm(m->x, m->rms1[l], m->xb, d);
        matvec(&m->wq[l], m->xb, m->q);
        matvec(&m->wk[l], m->xb, m->k);
        matvec(&m->wv[l], m->xb, m->v);

        float *kc = m->k_cache + ((size_t)l * ctx + pos) * d;
        float *vc = m->v_cache + ((size_t)l * ctx + pos) * d;
        memcpy(kc, m->k, sizeof(float) * d);
        memcpy(vc, m->v, sizeof(float) * d);

        const float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < nh; h++) {
            const float *qh = m->q + h * hd;
            float maxv = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const float *kt = m->k_cache + ((size_t)l * ctx + t) * d + h * hd;
                float s = 0.0f;
                for (int j = 0; j < hd; j++) s += qh[j] * kt[j];
                s *= scale;
                m->scores[t] = s;
                if (s > maxv) maxv = s;
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float e = expf(m->scores[t] - maxv);
                m->scores[t] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < hd; j++) {
                float acc = 0.0f;
                for (int t = 0; t <= pos; t++)
                    acc += m->scores[t] * m->v_cache[((size_t)l * ctx + t) * d + h * hd + j];
                m->att[h * hd + j] = acc * inv;
            }
        }
        matvec(&m->wo[l], m->att, m->xb);
        for (int i = 0; i < d; i++) m->x[i] += m->xb[i];

        rmsnorm(m->x, m->rms2[l], m->xb, d);
        matvec(&m->w1[l], m->xb, m->hb);
        for (int i = 0; i < mlp; i++) m->hb[i] = silu(m->hb[i]);
        matvec(&m->w2[l], m->hb, m->xb);
        for (int i = 0; i < d; i++) m->x[i] += m->xb[i];
    }

    rmsnorm(m->x, m->rms_f, m->xb, d);
    for (int v = 0; v < m->c.vocab; v++) m->logits[v] = row_dot(&m->tok_emb, v, m->xb);
    return 0.0f;
}

static uint32_t next_rand(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

int eslm_sample(eslm_t *m, float temp, int top_k) {
    int V = m->c.vocab;
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; i++)
            if (m->logits[i] > m->logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < V; i++) m->logits[i] /= temp;

    int k = (top_k > 0 && top_k < V) ? top_k : V;
    float thresh = -1e30f;
    for (int r = 0; r < k; r++) {
        float mx = -1e30f;
        for (int i = 0; i < V; i++)
            if (m->logits[i] > mx && (r == 0 || m->logits[i] < thresh)) mx = m->logits[i];
        thresh = mx;
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        m->probs[i] = (m->logits[i] >= thresh) ? expf(m->logits[i] - thresh) : 0.0f;
        sum += m->probs[i];
    }
    float r = (float)(next_rand(&m->rng) >> 8) / (float)(1u << 24) * sum;
    float acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += m->probs[i];
        if (acc >= r) return i;
    }
    return V - 1;
}

size_t eslm_token_bytes(const eslm_t *m, int id, const uint8_t **ptr) {
    if (!m->tok_offs || id < 0 || id >= m->c.vocab) {
        if (ptr) *ptr = NULL;
        return 0;
    }
    uint32_t a = m->tok_offs[id], b = m->tok_offs[id + 1];
    if (b < a) b = a;
    if (ptr) *ptr = m->tok_blob + a;
    return (size_t)(b - a);
}

int eslm_special_e(const eslm_t *m) { return m->sp_e; }

int eslm_is_special(const eslm_t *m, int id) {
    return id == m->sp_u || id == m->sp_a || id == m->sp_e;
}
