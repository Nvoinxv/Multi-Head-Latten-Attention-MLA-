# Multi-Head Latent Attention (MLA)

> Implementasi PyTorch untuk **Multi-Head Latent Attention**, mekanisme attention efisien memori yang diperkenalkan pada arsitektur DeepSeek-V2, dilengkapi **Decoupled Rotary Positional Embedding (RoPE)** dan integrasi dengan **Mixture-of-Experts (MoE) Feed Forward**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Educational%20Implementation-yellow)

---

## Daftar Isi

1. [Latar Belakang](#latar-belakang)
2. [Masalah yang Diselesaikan MLA](#masalah-yang-diselesaikan-mla)
3. [Arsitektur](#arsitektur)
4. [Formulasi Matematis](#formulasi-matematis)
5. [Decoupled RoPE](#decoupled-rope)
6. [Struktur Model Lengkap](#struktur-model-lengkap)
7. [Instalasi](#instalasi)
8. [Penggunaan](#penggunaan)
9. [Struktur Proyek](#struktur-proyek)
10. [Perbandingan dengan MHA](#perbandingan-dengan-mha)
11. [Limitasi & Catatan Jujur](#limitasi--catatan-jujur)
12. [Referensi](#referensi)
13. [Lisensi](#lisensi)

---

## Latar Belakang

**Multi-Head Latent Attention (MLA)** adalah varian mekanisme *self-attention* yang dirancang untuk mengatasi salah satu bottleneck utama pada inference model bahasa skala besar: **ukuran KV cache**. Pada Multi-Head Attention (MHA) standar, setiap token menyimpan vektor Key ($K$) dan Value ($V$) penuh untuk seluruh head, sehingga memori yang dibutuhkan tumbuh linear terhadap panjang urutan, jumlah head, dan dimensi per head.

MLA mengatasi hal ini dengan memproyeksikan $K$ dan $V$ ke dalam sebuah **vektor laten berdimensi rendah** sebelum disimpan, sehingga hanya representasi terkompresi inilah yang perlu di-cache — bukan $K$ dan $V$ untuk setiap head secara terpisah.

---

## Masalah yang Diselesaikan MLA

| Masalah pada MHA Standar | Solusi pada MLA |
|---|---|
| KV cache tumbuh $O(n_h \cdot d_h)$ per token | Kompresi ke latent, KV cache $O(d_c)$ per token |
| RoPE tidak kompatibel dengan kompresi low-rank | Decoupled RoPE — rotasi hanya pada sub-komponen kecil |
| Trade-off antara kapasitas model dan efisiensi memori | Kapasitas representasi tetap terjaga lewat up-projection |

---

## Arsitektur

Posisi MLA dalam satu block Transformer (pre-norm, dengan residual connection):

```
                    ┌────────────────────────────────────┐
                    │              Input h_t               │
                    └───────────────────┬──────────────────┘
                                         │
                                    RMSNorm
                                         │
                          ┌──────────────┴──────────────┐
                          │   Multi-Head Latent Attention │
                          │                                │
                          │   h_t ──► c_KV (down-proj)      │
                          │   c_KV ──► K_C, V_C (up-proj)  │
                          │   h_t ──► K_R ──► RoPE           │
                          │   K = [K_C ; K_R]              │
                          │   Q dibentuk serupa dari c_Q    │
                          │                                │
                          │   softmax(QK^T/√d) V           │
                          └──────────────┬──────────────┘
                                         │
                                    (+) residual dari h_t
                                         │
                                    RMSNorm
                                         │
                                   MoE Feed Forward
                                         │
                                    (+) residual
                                         │
                                    Output block
```

---

## Formulasi Matematis

### 1. Kompresi Low-Rank Key & Value

Hidden state $\mathbf{h}_t \in \mathbb{R}^{d_{\text{model}}}$ diproyeksikan ke vektor laten bersama $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$, dengan $d_c \ll n_h \cdot d_h$:

$$
\mathbf{c}_t^{KV} = W^{DKV}\, \mathbf{h}_t
$$

Key dan Value direkonstruksi dari latent yang sama:

$$
\mathbf{k}_t^{C} = W^{UK}\, \mathbf{c}_t^{KV}, \qquad
\mathbf{v}_t^{C} = W^{UV}\, \mathbf{c}_t^{KV}
$$

Hanya $\mathbf{c}_t^{KV}$ yang perlu disimpan dalam KV cache — bukan $\mathbf{k}_t^{C}$ maupun $\mathbf{v}_t^{C}$ secara penuh.

### 2. Kompresi Query (opsional, untuk efisiensi training)

$$
\mathbf{c}_t^{Q} = W^{DQ}\, \mathbf{h}_t, \qquad
\mathbf{q}_t^{C} = W^{UQ}\, \mathbf{c}_t^{Q}
$$

### 3. Skor Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_h + d_h^R}}\right) V
$$

dengan $d_h^R$ adalah dimensi komponen rotary tambahan (dijelaskan pada bagian berikut).

---

## Decoupled RoPE

RoPE merotasi $Q$ dan $K$ berdasarkan posisi **sebelum** dot-product dihitung. Masalahnya, jika RoPE diterapkan langsung pada $\mathbf{k}_t^{C} = W^{UK} \mathbf{c}_t^{KV}$, matriks rotasi $R_{pos}$ **tidak dapat dilebur** ke dalam $W^{UK}$ karena rotasi bergantung pada posisi $t$, sementara $W^{UK}$ bersifat statis — kompresi low-rank kehilangan efisiensinya.

**Solusi:** komponen posisi dipisah (*decoupled*) dari komponen konten. Sebuah key rotary tambahan berdimensi kecil diproyeksikan **langsung dari $\mathbf{h}_t$** (bukan dari latent) dan **dibagikan ke seluruh head**:

$$
\mathbf{k}_t^{R} = \text{RoPE}\big(W^{KR}\, \mathbf{h}_t\big)
$$

$$
\mathbf{k}_t = \big[\, \mathbf{k}_t^{C} \; ; \; \mathbf{k}_t^{R} \,\big]
$$

Query mengikuti pola serupa, dengan komponen rotary dihitung per-head dari latent Query:

$$
\mathbf{q}_t^{R} = \text{RoPE}\big(W^{QR}\, \mathbf{c}_t^{Q}\big), \qquad
\mathbf{q}_t = \big[\, \mathbf{q}_t^{C} \; ; \; \mathbf{q}_t^{R} \,\big]
$$

Sifat kunci RoPE tetap terjaga: hasil dot-product antara $Q$ di posisi $m$ dan $K$ di posisi $n$ hanya bergantung pada selisih relatif $(m-n)$:

$$
\langle f_q(x_m, m),\, f_k(x_n, n) \rangle = g(x_m, x_n, m-n)
$$

---

## Struktur Model Lengkap

```
Token IDs
   │
Embedding
   │
┌───────────── × N layer ─────────────┐
│  RMSNorm → MLA (RoPE di dalamnya)    │
│    │                                  │
│  (+) residual                         │
│    │                                  │
│  RMSNorm → MoE Feed Forward           │
│    │                                  │
│  (+) residual                         │
└────────────────────────────────────────┘
   │
RMSNorm (final)
   │
Linear (proyeksi ke vocab)
   │
Logits → Softmax (pada loss function)
```

> RoPE **tidak** diterapkan pada embedding token secara langsung. RoPE hanya bekerja pada sub-komponen rotary $Q$/$K$ di dalam setiap layer MLA.

---

## Instalasi

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install torch
```

---

## Penggunaan

```python
import torch
from model import GPTStyleMoEModel

model = GPTStyleMoEModel(
    vocab_size=32000,
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_head=64,
    d_rotary=32,
    d_latent_kv=128,
    d_latent_q=192,
    d_ffn_hidden=1024,
    n_routed_experts=8,
    n_shared_experts=1,
    top_k=2,
    max_seq_len=2048,
)

token_ids = torch.randint(0, 32000, (4, 16))   # [batch, seq_len]
logits, aux_loss = model(token_ids)

print(logits.shape)   # [4, 16, 32000]
```

---

## Struktur Proyek

```
.
├── model.py              # GPTStyleMoEModel (embedding + stack layer + output head)
├── mla.py                 # MultiHeadLatentAttention
├── moe.py                  # Expert, MoEFeedForward
├── rope.py                  # RotaryPositionEmbedding
├── norm.py                   # RMSNorm
├── block.py                   # TransformerBlock
├── train.py                    # Training loop
└── README.md
```

---

## Perbandingan dengan MHA

| Aspek | MHA Standar | MLA |
|---|---|---|
| Yang disimpan di cache | $K, V$ penuh per head | Latent $\mathbf{c}_t^{KV}$ + $\mathbf{k}_t^{R}$ (kecil) |
| Ukuran KV cache per token | $O(n_h \cdot d_h)$ | $O(d_c + d_h^R)$ |
| Informasi posisi | Menyatu dengan $K$ penuh | Dipisah (*decoupled*) dari komponen konten |
| Kompatibilitas kompresi low-rank | Tidak relevan | Terjaga karena rotasi dipisah dari konten |

---

## Limitasi & Catatan Jujur

Implementasi pada repositori ini ditujukan untuk **tujuan pembelajaran**, bukan optimisasi produksi. Beberapa perbedaan penting dari implementasi produksi (mis. DeepSeek-V2 resmi):

- **Belum ada matrix absorption saat inference.** Pada implementasi produksi, $W^{UK}$ dapat "dilebur" ke dalam $W^{UQ}$ sehingga $K$ tidak perlu direkonstruksi penuh dari latent saat inference — cukup latent yang di-cache dan dipakai langsung dalam bentuk terkompresi. Implementasi ini masih merekonstruksi $\mathbf{k}_t^{C}$ dan $\mathbf{v}_t^{C}$ secara penuh pada setiap forward pass, sehingga penghematan memori KV cache belum sepenuhnya termanfaatkan saat inference autoregresif.
- **Routing MoE menggunakan loop Python per-expert**, bukan operasi *scatter/gather* batched atau kernel khusus. Cukup untuk skala kecil dan tujuan edukasi, tapi tidak efisien untuk jumlah expert atau batch besar.
- **Belum ada KV cache eksplisit** untuk inference token-by-token (generation). Forward pass saat ini mengasumsikan seluruh urutan diproses sekaligus (cocok untuk training, belum untuk inference streaming).

---

## Referensi

- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*
- Su, J. et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Vaswani, A. et al., *Attention Is All You Need*

---

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).