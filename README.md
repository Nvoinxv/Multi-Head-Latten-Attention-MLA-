# 🚀 Multi-Head Latent Attention (MLA) & Mixture-of-Experts (MoE)

> Implementasi PyTorch komprehensif untuk **Multi-Head Latent Attention (MLA)**. Arsitektur ini mengadopsi mekanisme attention efisien memori yang diperkenalkan pada model **DeepSeek-V2**, dilengkapi dengan **Decoupled Rotary Positional Embedding (RoPE)** dan terintegrasi penuh dengan **Mixture-of-Experts (MoE) Feed Forward Network**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Educational%20Implementation-yellow)

---

## 📑 Daftar Isi

1. [Latar Belakang](#latar-belakang)
2. [Masalah yang Diselesaikan MLA](#masalah-yang-diselesaikan-mla)
3. [Arsitektur Keseluruhan](#arsitektur-keseluruhan)
4. [Formulasi Matematis (LaTeX)](#formulasi-matematis-latex)
5. [Decoupled RoPE](#decoupled-rope)
6. [Struktur Model Lengkap](#struktur-model-lengkap)
7. [Instalasi](#instalasi)
8. [Penggunaan Cepat](#penggunaan-cepat)
9. [Struktur Proyek](#struktur-proyek)
10. [Perbandingan MLA vs MHA](#perbandingan-mla-vs-mha)
11. [Limitasi & Catatan Pengembangan](#limitasi--catatan-pengembangan)
12. [Referensi](#referensi)
13. [Lisensi](#lisensi)

---

## 🌟 Latar Belakang

**Multi-Head Latent Attention (MLA)** adalah varian inovatif dari mekanisme *self-attention* yang secara spesifik dirancang untuk mengatasi *bottleneck* utama pada inferensi Large Language Models (LLM): **ukuran *Key-Value (KV) cache***.

Pada mekanisme *Multi-Head Attention (MHA)* standar, setiap token harus menyimpan representasi vektor penuh dari **Key** ($K$) dan **Value** ($V$) untuk seluruh *attention heads*. Akibatnya, alokasi memori akan tumbuh secara linear terhadap:
1. Panjang urutan (*sequence length*).
2. Jumlah *heads* ($n_h$).
3. Dimensi per *head* ($d_h$).

**Solusi MLA:** Mekanisme ini memproyeksikan $K$ dan $V$ ke dalam sebuah **vektor laten berdimensi rendah (*low-dimensional latent vector*)** sebelum disimpan ke dalam *cache*. Dengan demikian, sistem hanya perlu menyimpan representasi terkompresi ini, bukan vektor $K$ dan $V$ secara terpisah untuk setiap *head*, sehingga menghemat penggunaan memori (VRAM) secara signifikan.

---

## 🎯 Masalah yang Diselesaikan MLA

| Tantangan pada MHA Standar | Solusi Inovatif pada MLA |
| :--- | :--- |
| Pertumbuhan KV cache sebesar $\mathcal{O}(n_h \cdot d_h)$ per token | Kompresi laten, menjadikan KV cache hanya $\mathcal{O}(d_c)$ per token |
| Dekomposisi *low-rank* merusak sifat RoPE konvensional | Menggunakan **Decoupled RoPE**, memisahkan rotasi ke sub-komponen khusus |
| Pertukaran (*trade-off*) antara efisiensi memori & kapasitas model | Kapasitas dijaga ketat melalui mekanisme *up-projection* yang presisi |

---

## 🏗️ Arsitektur Keseluruhan

Posisi komponen MLA dalam satu *Transformer Block* (menggunakan arsitektur *pre-norm* dengan *residual connections*):

```text
                    ┌────────────────────────────────────┐
                    │            Input token (h_t)       │
                    └───────────────────┬──────────────────┘
                                        ▼
                                   [RMSNorm]
                                        ▼
                          ┌─────────────┴─────────────┐
                          │ Multi-Head Latent Attention │
                          │                             │
                          │ h_t ──► c_KV (down-proj)    │
                          │ c_KV ──► K_C, V_C (up-proj) │
                          │ h_t ──► K_R ──► RoPE        │
                          │ K = [K_C ; K_R]             │
                          │ (Q diproses serupa)         │
                          │                             │
                          │ softmax(Q K^T / √d) V       │
                          └─────────────┬─────────────┘
                                        ▼
                                (+) Residual Add
                                        ▼
                                   [RMSNorm]
                                        ▼
                              [MoE Feed Forward]
                                        ▼
                                (+) Residual Add
                                        ▼
                                  Output block
```

---

## 📐 Formulasi Matematis (LaTeX)

Bagian ini mendefinisikan operasi matematis utama menggunakan notasi LaTeX standar.

### 1. Kompresi Low-Rank untuk Key & Value

Misalkan representasi *hidden state* pada token ke-$t$ adalah $\mathbf{h}_t \in \mathbb{R}^{d_{\mathrm{model}}}$. Representasi ini diproyeksikan ke dalam sebuah vektor laten bersama (*joint latent vector*) $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$, di mana dimensi laten $d_c$ jauh lebih kecil daripada dimensi total MHA ($d_c \ll n_h \cdot d_h$):

$$
\mathbf{c}_t^{KV} = W^{DKV} \, \mathbf{h}_t
$$

Komponen *Key* ($\mathbf{k}_t^{C}$) dan *Value* ($\mathbf{v}_t^{C}$) direkonstruksi langsung dari vektor laten bersama ini menggunakan matriks *up-projection*:

$$
\mathbf{k}_t^{C} = W^{UK} \, \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^{C} = W^{UV} \, \mathbf{c}_t^{KV}
$$

> **💡 Inti Efisiensi:** Pada tahap inferensi (*generation*), sistem **hanya** menyimpan $\mathbf{c}_t^{KV}$ dalam memori (KV Cache), mereduksi penggunaan memori secara drastis.

### 2. Kompresi Query (Opsional untuk Pelatihan)

Untuk mengurangi kompleksitas memori saat fase pelatihan (*training*), hal yang sama juga dapat diterapkan pada *Query*:

$$
\mathbf{c}_t^{Q} = W^{DQ} \, \mathbf{h}_t, \quad \mathbf{q}_t^{C} = W^{UQ} \, \mathbf{c}_t^{Q}
$$

### 3. Matriks Attention

Fungsi *attention* dihitung melalui *scaled dot-product* biasa, ditambah dengan dimensi rotasi (*rotary component*):

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_h + d_h^R}}\right) V
$$

*Dimana $d_h^R$ merepresentasikan jumlah dimensi yang dialokasikan khusus untuk informasi rotasi posisional (RoPE).*

---

## 🔄 Decoupled RoPE

Rotary Positional Embedding (RoPE) merotasi komponen vektor berdasarkan posisinya *sebelum* dot-product.
Namun, apabila RoPE diterapkan secara langsung pada vektor yang direkonstruksi dari laten ($\mathbf{k}_t^{C}$), komputasi matriks *up-projection* ($W^{UK}$) menjadi sangat tidak efisien karena harus dieksekusi terus-menerus secara dinamis terhadap posisi token $t$.

**Solusi (Decoupled RoPE):**
Informasi posisional **dipisahkan (decoupled)** dari komponen semantik utama. Sebuah *Key Rotary* berdimensi kecil ($\mathbf{k}_t^{R}$) diciptakan **langsung dari input** ($\mathbf{h}_t$), dilewatkan ke dalam RoPE, dan disebarkan (*broadcasted*) ke seluruh *attention heads*:

$$
\mathbf{k}_t^{R} = \mathrm{RoPE}\Big(W^{KR} \, \mathbf{h}_t\Big)
$$

Komponen *Key* akhir merupakan gabungan (*concatenation*) antara komponen semantik laten dengan komponen rotary:

$$
\mathbf{k}_t = \Big[\, \mathbf{k}_t^{C} \; \Vert \; \mathbf{k}_t^{R} \,\Big]
$$

Polanya serupa untuk *Query*:

$$
\mathbf{q}_t^{R} = \mathrm{RoPE}\Big(W^{QR} \, \mathbf{c}_t^{Q}\Big), \quad \mathbf{q}_t = \Big[\, \mathbf{q}_t^{C} \; \Vert \; \mathbf{q}_t^{R} \,\Big]
$$

Hasil akhir menjamin bahwa RoPE tetap mempertahankan sifat matematis utamanya di mana perhitungan skor dependensi murni didasarkan pada selisih jarak relatif antar token $(m - n)$:

$$
\langle f_q(x_m, m),\, f_k(x_n, n) \rangle = g(x_m, x_n, m-n)
$$

---

## 🧩 Struktur Model Lengkap

Alur data model ini dari hulu ke hilir:

```text
[Input] Token IDs
       │
[Layer] Token Embedding
       │
┌──────┴────── [Transformer Block x N] ──────┐
│ 1. RMSNorm                                 │
│ 2. Multi-Head Latent Attention (w/ RoPE)   │
│ 3. Residual Add                            │
│ 4. RMSNorm                                 │
│ 5. Mixture-of-Experts (MoE) Feed Forward   │
│ 6. Residual Add                            │
└──────────────────────┬─────────────────────┘
                       │
[Layer] RMSNorm (Final)
                       │
[Layer] Linear Output (Vocab Projection)
                       │
[Output] Logits ──► Softmax (CrossEntropyLoss)
```

---

## 💻 Instalasi

Untuk mencoba implementasi ini secara lokal, pastikan Anda menggunakan lingkungan Python terbaru dan pustaka PyTorch.

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirement.txt
```

---

## 🚀 Penggunaan Cepat

Berikut adalah contoh inisiasi model *GPT-Style* dengan dukungan MoE:

```python
import torch
from model import GPTStyleMoEModel

# Konfigurasi Model
model = GPTStyleMoEModel(
    vocab_size=32000,
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_head=64,
    d_rotary=32,            # Dimensi Decoupled RoPE
    d_latent_kv=128,        # Kompresi laten untuk K/V
    d_latent_q=192,         # Kompresi laten untuk Q
    d_ffn_hidden=1024,
    n_routed_experts=8,     # Total Expert MoE
    n_shared_experts=1,     # Expert Umum
    top_k=2,                # Routing 2 expert terbaik per token
    max_seq_len=2048,
)

# Dummy Input Data [Batch, Sequence_Length]
token_ids = torch.randint(0, 32000, (4, 16))

# Forward Pass
logits, aux_loss = model(token_ids)

print(f"Bentuk Output Logits: {logits.shape}") 
# Output: torch.Size([4, 16, 32000])
```

*(Lihat notebook `AI_Transfomers.ipynb` untuk alur training autoregresif dan inference/prompting yang lengkap).*

---

## 📁 Struktur Proyek

```text
.
├── model.py              # Wrapper arsitektur utama model
├── mla.py                # Mekanisme Multi-Head Latent Attention
├── moe.py                # MoE Feed Forward & Expert Router
├── rope.py               # Algoritma Rotary Position Embedding
├── norm.py               # Lapisan Root Mean Square Normalization
├── block.py              # Definisi TransformerBlock
├── train.py              # Skrip eksekusi pelatihan
├── AI_Transfomers.ipynb  # Notebook panduan lengkap (Kode dari A-Z)
└── README.md             # Dokumentasi ini
```

---

## ⚖️ Perbandingan MLA vs MHA

| Atribut Pengukuran | MHA (Standar) | MLA (DeepSeek-V2) |
| :--- | :--- | :--- |
| **Penyimpanan Cache** | Vektor $K, V$ secara penuh per *head* | Vektor laten $\mathbf{c}_t^{KV} + \mathbf{k}_t^{R}$ |
| **Beban Memori Cache** | $\mathcal{O}(n_h \cdot d_h)$ per token | $\mathcal{O}(d_c + d_h^R)$ per token |
| **Pemrosesan Posisi** | Digabungkan secara inheren pada vektor $K$ | Dipisahkan (*Decoupled*) dari semantik |
| **Efisiensi Low-Rank** | Tidak relevan | Optimal, komputasi terkompresi |

---

## ⚠️ Limitasi & Catatan Pengembangan

Implementasi dalam repositori ini ditujukan sebagai **materi pembelajaran (educational purpose)** dan bukan *production-grade framework*. Beberapa abstraksi telah disederhanakan:

1. **Belum Ada Matrix Absorption saat Inferensi:** 
   Pada model skala produksi seperti DeepSeek, $W^{UK}$ dilebur (*absorbed*) langsung ke dalam matriks *up-projection* saat inferensi, meniadakan perlunya rekonstruksi penuh vektor $K$. Implementasi ini masih melakukan komputasi rekonstruksi pada *forward pass* untuk alasan keterbacaan kode.
2. **MoE Routing Sederhana:** 
   Proses *routing* MoE pada *loop* Python saat ini belum memanfaatkan kompilasi kernel *scatter/gather* di tingkat C++/CUDA, sehingga belum *optimized* untuk ukuran batch/expert raksasa.
3. **Generasi Autoregresif Dasar:** 
   Fungsi KV-Cache yang statis dan optimal untuk kecepatan respons *streaming* masih dapat ditingkatkan lagi pemanfaatannya (saat ini *generate()* diimplementasikan melalui proses agregasi konteks yang standar).

---

## 📚 Referensi

1. DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.*
2. Su, Jianlin et al. (2021). *RoFormer: Enhanced Transformer margin with Rotary Position Embedding.*
3. Vaswani, Ashish et al. (2017). *Attention Is All You Need.*

---

## 📜 Lisensi

Kode dan dokumentasi proyek ini dilisensikan di bawah spesifikasi [MIT License](LICENSE).