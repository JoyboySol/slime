MODEL_ARGS=(
   --custom-model-provider-path slime_plugins.models.yulan_moe.model_provider

   --disable-bias-linear
   --add-qkv-bias
   --attention-output-gate
   --gated-norm
   --gated-norm-lora-rank 16
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
   --vocab-size 99072
   --max-position-embeddings 131072
   --position-embedding-type none
   --final-logits-clamp-value 30
   --activation-func-clamp-value 7
   --activation-func-clamp-mode soft

   --is-hybrid-model
   # SFT uses the GPT builder, where linear_attention_freq/moe_layer_freq set
   # each layer's spec. Slime's YuLan provider uses MambaStack.allocate_layers;
   # keep its auto-ratios at zero so the exact 49-GDN/7-attention override below
   # is authoritative without the allocator enforcing unrelated auto-counts.
   # Per-layer dense/MoE selection still comes from moe_layer_freq.
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
   --gdn-split-in-proj
   --gdn-fused-split-forward
   --gdn-alpha-weight-scale 1.0
   --gdn-alpha-weight-scale-mode matrix
   --gdn-lr-mult-alpha 0.5

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
   --moe-aux-loss-coeff 0.05
   --moe-router-load-balancing-type global_aux_loss
)
