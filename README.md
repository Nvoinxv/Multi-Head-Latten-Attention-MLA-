Dataset
  ↓
Tokenizer → Token IDs → Tensor PyTorch
  ↓
Embedding
  ↓
┌─── (ulangi N kali) ──────────────────────┐
│  RMSNorm                                  │
│    ↓                                      │
│  Multi-Head Latent Attention (RoPE di dalamnya)
│    ↓                                      │
│  + residual (dari input block)            │
│    ↓                                      │
│  RMSNorm                                  │
│    ↓                                      │
│  MoE Feed Forward                         │
│    ↓                                      │
│  + residual (dari sebelum MoE)             │
└────────────────────────────────────────────┘
  ↓
RMSNorm (final, sekali saja setelah semua block)
  ↓
Linear
  ↓
Softmax