/*
 * ssm_weights.c — Weight Loader for Mamba2 SSM Bare Metal
 *
 * Loads .mamba.bin v2 files into MambaWeights struct.
 * Zero-copy: weight pointers point directly into the memory-mapped blob.
 */

#include "ssm_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Header Parsing ───────────────────────────────────────────────────────── */

int mamba_parse_header(
    MambaBinHeader *hdr,
    const void     *data,
    uint64_t        data_len
)
{
    /**
     * Parse .mamba.bin header from raw bytes.
     * Validates magic and version.
     */
    if (!hdr || !data || data_len < sizeof(MambaBinHeader)) return -1;

    memcpy(hdr, data, sizeof(MambaBinHeader));

    if (hdr->magic != MAMBA_BIN_MAGIC) {
        return -1;
    }
    if (hdr->version != MAMBA_BIN_VERSION && hdr->version != 1) {
        return -1;
    }
    if (hdr->d_model <= 0 || hdr->n_layers <= 0 || hdr->vocab_size <= 0) {
        return -1;
    }
    return 0;
}

/* ── Weight Loading ───────────────────────────────────────────────────────── */

/**
 * Advance cursor pointer and return float* at current position.
 */
static float *claim_floats(const uint8_t **cursor, uint64_t *remaining, int64_t count)
{
    /**
     * Claim a contiguous block of float32 values from the data stream.
     */
    if (count <= 0) return NULL;
    uint64_t bytes = (uint64_t)count * sizeof(float);
    if (bytes > *remaining) return NULL;

    float *ptr = (float *)(*cursor);
    *cursor    += bytes;
    *remaining -= bytes;
    return ptr;
}

int mamba_load_weights(
    MambaWeights       *wt,
    MambaConfig        *cfg,
    const void         *data,
    uint64_t            data_len
)
{
    /**
     * Load all Mamba2 weights from .mamba.bin v2 buffer.
     *
     * Per-layer tensor layout (written by export_mamba_baremetal.py):
     *   layer_norm      [d_model]
     *   in_proj         [in_proj_dim * d_model]
     *   conv1d_weight   [conv_dim * d_conv]
     *   conv1d_bias     [conv_dim]
     *   inner_norm      [d_inner]
     *   out_proj        [d_model * d_inner]
     *   A_log           [nheads]
     *   D               [nheads]
     *   dt_bias         [nheads]
     */
    if (!wt || !cfg || !data) return -1;
    memset(wt, 0, sizeof(*wt));

    MambaBinHeader hdr;
    if (mamba_parse_header(&hdr, data, data_len) != 0) return -1;

    /* Fill config from header */
    cfg->d_model       = hdr.d_model;
    cfg->d_state       = hdr.d_state;
    cfg->d_conv        = hdr.d_conv;
    cfg->expand        = hdr.expand;
    cfg->d_inner       = hdr.d_model * hdr.expand;
    cfg->n_layers      = hdr.n_layers;
    cfg->vocab_size    = hdr.vocab_size;
    cfg->max_seq_len   = hdr.max_seq_len;
    cfg->base_split    = hdr.base_split;
    cfg->max_rlf_loops = hdr.max_rlf_loops;
    cfg->halt_token_id = hdr.halt_token_id;
    cfg->rope_base     = hdr.rope_base;

    /* Mamba2 multi-head fields */
    cfg->nheads        = hdr.nheads > 0 ? hdr.nheads : 80;
    cfg->headdim       = hdr.headdim > 0 ? hdr.headdim : 64;
    cfg->ngroups       = hdr.ngroups > 0 ? hdr.ngroups : 1;
    cfg->conv_dim      = cfg->d_inner + 2 * cfg->ngroups * cfg->d_state;
    cfg->in_proj_dim   = 2 * cfg->d_inner + 2 * cfg->ngroups * cfg->d_state + cfg->nheads;
    cfg->prefix_m      = hdr.prefix_m;
    cfg->bridge_rank   = hdr.bridge_rank;
    cfg->loop_nheads   = hdr.loop_nheads > 0 ? hdr.loop_nheads : 20;
    cfg->loop_headdim  = hdr.loop_headdim > 0 ? hdr.loop_headdim : 128;
    cfg->loop_d_state  = hdr.loop_d_state > 0 ? hdr.loop_d_state : 32;
    cfg->loop_d_inner  = cfg->d_model;  /* expand=1 for loop core */
    cfg->loop_conv_dim = cfg->loop_d_inner + 2 * cfg->loop_d_state;

    int d_model    = hdr.d_model;
    int d_inner    = cfg->d_inner;
    int d_conv     = hdr.d_conv;
    int n_layers   = hdr.n_layers;
    int vocab      = hdr.vocab_size;
    int nheads     = cfg->nheads;
    int conv_dim   = cfg->conv_dim;
    int in_proj_d  = cfg->in_proj_dim;
    int prefix_m   = cfg->prefix_m;
    int bridge_rank = cfg->bridge_rank;
    int loop_d_inner  = cfg->loop_d_inner;
    int loop_nheads   = cfg->loop_nheads;
    int loop_conv_dim = cfg->loop_conv_dim;
    int loop_d_state  = cfg->loop_d_state;
    int loop_in_proj_d = 2 * loop_d_inner + 2 * loop_d_state + loop_nheads;

    /* Allocate per-layer pointer arrays */
    wt->layer_norm    = (float **)calloc(n_layers, sizeof(float *));
    wt->in_proj       = (float **)calloc(n_layers, sizeof(float *));
    wt->conv1d_weight = (float **)calloc(n_layers, sizeof(float *));
    wt->conv1d_bias   = (float **)calloc(n_layers, sizeof(float *));
    wt->inner_norm    = (float **)calloc(n_layers, sizeof(float *));
    wt->out_proj      = (float **)calloc(n_layers, sizeof(float *));
    wt->A_log         = (float **)calloc(n_layers, sizeof(float *));
    wt->D             = (float **)calloc(n_layers, sizeof(float *));
    wt->dt_bias       = (float **)calloc(n_layers, sizeof(float *));

    if (!wt->layer_norm || !wt->in_proj) {
        mamba_free_weight_arrays(wt);
        return -1;
    }

    /* Walk through binary data after header */
    const uint8_t *cursor   = (const uint8_t *)data + sizeof(MambaBinHeader);
    uint64_t       remaining = data_len - sizeof(MambaBinHeader);

    /* Global tensors */
    wt->token_embedding  = claim_floats(&cursor, &remaining, (int64_t)vocab * d_model);
    wt->lm_head          = claim_floats(&cursor, &remaining, (int64_t)vocab * d_model);
    wt->final_norm_weight = claim_floats(&cursor, &remaining, d_model);

    if (!wt->token_embedding || !wt->lm_head || !wt->final_norm_weight) {
        mamba_free_weight_arrays(wt);
        return -1;
    }

    /* Per-layer tensors (Mamba2 layout) */
    for (int l = 0; l < n_layers; l++) {
        wt->layer_norm[l]    = claim_floats(&cursor, &remaining, d_model);
        wt->in_proj[l]       = claim_floats(&cursor, &remaining, (int64_t)in_proj_d * d_model);
        wt->conv1d_weight[l] = claim_floats(&cursor, &remaining, (int64_t)conv_dim * d_conv);
        wt->conv1d_bias[l]   = claim_floats(&cursor, &remaining, conv_dim);
        wt->inner_norm[l]    = claim_floats(&cursor, &remaining, d_inner);
        wt->out_proj[l]      = claim_floats(&cursor, &remaining, (int64_t)d_model * d_inner);
        wt->A_log[l]         = claim_floats(&cursor, &remaining, nheads);
        wt->D[l]             = claim_floats(&cursor, &remaining, nheads);
        wt->dt_bias[l]       = claim_floats(&cursor, &remaining, nheads);

        if (!wt->layer_norm[l] || !wt->in_proj[l]) {
            mamba_free_weight_arrays(wt);
            return -1;
        }
    }

    /* RLF weights (optional) */
    if (hdr.has_rlf) {
        wt->lifeline_gate       = claim_floats(&cursor, &remaining, d_model);
        wt->loop_norm_weight    = claim_floats(&cursor, &remaining, d_model);
        wt->loop_in_proj        = claim_floats(&cursor, &remaining, (int64_t)loop_in_proj_d * d_model);
        wt->loop_conv1d_weight  = claim_floats(&cursor, &remaining, (int64_t)loop_conv_dim * d_conv);
        wt->loop_conv1d_bias    = claim_floats(&cursor, &remaining, loop_conv_dim);
        wt->loop_inner_norm     = claim_floats(&cursor, &remaining, loop_d_inner);
        wt->loop_out_proj       = claim_floats(&cursor, &remaining, (int64_t)d_model * loop_d_inner);
        wt->loop_A_log          = claim_floats(&cursor, &remaining, loop_nheads);
        wt->loop_D              = claim_floats(&cursor, &remaining, loop_nheads);
        wt->loop_dt_bias        = claim_floats(&cursor, &remaining, loop_nheads);
    }

    /* Prefix Latent Scratchpad */
    if (prefix_m > 0) {
        wt->latent_memory = claim_floats(&cursor, &remaining, (int64_t)prefix_m * d_model);
    }

    /* Latent Bridge */
    if (bridge_rank > 0) {
        wt->bridge_down = claim_floats(&cursor, &remaining, (int64_t)bridge_rank * d_model);
        wt->bridge_up   = claim_floats(&cursor, &remaining, (int64_t)d_model * bridge_rank);
    }

    /* Store blob reference */
    wt->_blob       = (float *)data;
    wt->_blob_bytes = data_len;

    return 0;
}

void mamba_free_weight_arrays(MambaWeights *wt)
{
    /**
     * Free per-layer pointer arrays (but NOT the backing data blob).
     */
    if (!wt) return;
    free(wt->layer_norm);     wt->layer_norm    = NULL;
    free(wt->in_proj);        wt->in_proj       = NULL;
    free(wt->conv1d_weight);  wt->conv1d_weight = NULL;
    free(wt->conv1d_bias);    wt->conv1d_bias   = NULL;
    free(wt->inner_norm);     wt->inner_norm    = NULL;
    free(wt->out_proj);       wt->out_proj      = NULL;
    free(wt->A_log);          wt->A_log         = NULL;
    free(wt->D);              wt->D             = NULL;
    free(wt->dt_bias);        wt->dt_bias       = NULL;
}

void mamba_print_summary(const MambaConfig *cfg, const MambaWeights *wt)
{
    /**
     * Print model loading summary.
     */
    if (!cfg) return;
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Mamba2-2.7B SSM + RLF Bare-Metal (v2)\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  d_model:       %d\n", cfg->d_model);
    printf("  d_state:       %d\n", cfg->d_state);
    printf("  d_conv:        %d\n", cfg->d_conv);
    printf("  expand:        %d\n", cfg->expand);
    printf("  d_inner:       %d\n", cfg->d_inner);
    printf("  n_layers:      %d\n", cfg->n_layers);
    printf("  vocab_size:    %d\n", cfg->vocab_size);
    printf("  nheads:        %d\n", cfg->nheads);
    printf("  headdim:       %d\n", cfg->headdim);
    printf("  ngroups:       %d\n", cfg->ngroups);
    printf("  conv_dim:      %d\n", cfg->conv_dim);
    printf("  in_proj_dim:   %d\n", cfg->in_proj_dim);
    printf("  base_split:    %d (RLF)\n", cfg->base_split);
    printf("  max_rlf_loops: %d\n", cfg->max_rlf_loops);
    printf("  halt_token_id: %d\n", cfg->halt_token_id);
    printf("  prefix_m:      %d\n", cfg->prefix_m);
    printf("  bridge_rank:   %d\n", cfg->bridge_rank);
    printf("  loop_nheads:   %d\n", cfg->loop_nheads);
    printf("  loop_headdim:  %d\n", cfg->loop_headdim);
    printf("  loop_d_state:  %d\n", cfg->loop_d_state);
    if (wt) {
        printf("  lifeline:      %s\n", wt->lifeline_gate ? "loaded" : "none");
        printf("  loop_core:     %s\n", wt->loop_in_proj ? "loaded" : "none");
        printf("  scratchpad:    %s\n", wt->latent_memory ? "loaded" : "none");
        printf("  bridge:        %s\n", wt->bridge_down ? "loaded" : "none");
        printf("  blob_bytes:    %llu\n", (unsigned long long)wt->_blob_bytes);
    }
    printf("══════════════════════════════════════════════════════════════\n\n");
}
