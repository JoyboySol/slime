MODEL_ARGS=(
   --custom-model-provider-path slime_plugins.models.yulan_moe.model_provider

   --disable-bias-linear
   --add-qkv-bias
   --disable-attn-output-gate
   --num-layers 56
   --hidden-size 2048
   --ffn-hidden-size 4800
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --kv-channels 64

   --normalization RMSNorm
   --norm-epsilon 1e-6
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 99000
   --max-position-embeddings 49152
   --position-embedding-type none

   --is-hybrid-model
   # The explicit pattern is authoritative. Zero ratios avoid the generic Mamba
   # allocator reinterpreting Qwen3-Next's per-layer MLPs as MLP-only layers.
   --hybrid-attention-ratio 0
   --hybrid-mlp-ratio 0
   --hybrid-override-pattern MMMMMMMMMMMM\*MMMMMMM\*\*M\*MMMMMMMMMMMMMMMMMMMMMM\*M\*\*MMMMMM
   --linear-attention-type gated_delta_net
   --linear-attention-freq "[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1]"
   --linear-conv-kernel-dim 4
   --linear-key-head-dim 64
   --linear-value-head-dim 64
   --linear-num-key-heads 8
   --linear-num-value-heads 32

   --num-experts 192
   --moe-layer-freq "[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
   --moe-ffn-hidden-size 512
   --moe-shared-expert-intermediate-size 512
   --moe-shared-expert-gate
   --moe-router-topk 4
   --moe-router-score-function softmax
   --moe-router-dtype fp32
   --moe-router-sqrt-gate
   --moe-grouped-gemm
)
