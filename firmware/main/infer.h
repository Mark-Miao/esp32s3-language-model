#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct eslm eslm_t;

// Parse an embedded "ESLM" blob (float32 weights only for now).
eslm_t *eslm_init(const uint8_t *blob, size_t len);
void eslm_deinit(eslm_t *m);

int eslm_ctx(const eslm_t *m);
int eslm_vocab(const eslm_t *m);
int eslm_params(const eslm_t *m);
int eslm_dtype(const eslm_t *m);

void eslm_reset(eslm_t *m);              // forget KV cache / RNG position
void eslm_set_seed(eslm_t *m, uint32_t seed);
void eslm_set_pos(eslm_t *m, int pos);   // after feeding a prompt

float eslm_forward(eslm_t *m, int token, int pos);  // fills logits, returns nothing
int eslm_sample(eslm_t *m, float temp, int top_k);

// v2 blobs carry the BPE vocabulary; decode a token id to raw bytes.
size_t eslm_token_bytes(const eslm_t *m, int id, const uint8_t **ptr);
int eslm_special_e(const eslm_t *m);
int eslm_is_special(const eslm_t *m, int id);
