# 🚀 NVOIN — Multi-Head Latent Attention (MLA) & Mixture-of-Experts (MoE) dari Nol

> Implementasi PyTorch edukatif untuk **Multi-Head Latent Attention (MLA)**, mekanisme attention efisien memori dari **DeepSeek-V2**, dilengkapi **Decoupled Rotary Positional Embedding (RoPE)** dan **Mixture-of-Experts (MoE) Feed Forward Network**. Seluruh implementasi — mulai dari load dataset, tokenisasi, arsitektur model, training, sampai inference — ada dalam **satu Jupyter Notebook**: `AI_Transfomers.ipynb`.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Educational%20Notebook-yellow)

> ⚠️ **Catatan struktur:** proyek ini **belum** berupa package/repo modular (tidak ada `model.py`, `mla.py`, `moe.py`, dll). Semua kode berada di dalam notebook. Bagian "Instalasi" dan "Penggunaan" di bawah disesuaikan dengan kondisi ini.

---

## 📑 Daftar Isi

1. [Latar Belakang](#latar-belakang)
2. [Masalah yang Diselesaikan MLA](#masalah-yang-diselesaikan-mla)
3. [Arsitektur Keseluruhan](#arsitektur-keseluruhan)
4. [Formulasi Matematis (LaTeX)](#formulasi-matematis-latex)
5. [Decoupled RoPE](#decoupled-rope)
6. [Struktur Notebook (Cell-by-Cell)](#struktur-notebook-cell-by-cell)
7. [Instalasi](#instalasi)
8. [Menjalankan Notebook](#menjalankan-notebook)
9. [Konfigurasi Model yang Dipakai](#konfigurasi-model-yang-dipakai)
10. [Struktur Proyek (Nyata)](#struktur-proyek-nyata)
11. [Perbandingan MLA vs MHA](#perbandingan-mla-vs-mha)
12. [Limitasi & Catatan Pengembangan](#limitasi--catatan-pengembangan)
13. [Referensi](#referensi)
14. [Lisensi](#lisensi)

---

## 🌟 Latar Belakang

**Multi-Head Latent Attention (MLA)** adalah varian *self-attention* yang dirancang untuk mengatasi *bottleneck* utama pada inferensi Large Language Models (LLM): **ukuran *Key-Value (KV) cache***.

Pada *Multi-Head Attention (MHA)* standar, setiap token menyimpan representasi $K$ dan $V$ penuh untuk seluruh *heads*, sehingga memori tumbuh linear terhadap panjang urutan, jumlah *heads* ($n_h$), dan dimensi per *head* ($d_h$).

**Solusi MLA:** memproyeksikan $K$ dan $V$ ke **vektor laten berdimensi rendah** sebelum disimpan ke cache, sehingga hanya representasi terkompresi yang perlu disimpan — bukan vektor $K$/$V$ terpisah per *head*.

---

## 🎯 Masalah yang Diselesaikan MLA

| Tantangan pada MHA Standar | Solusi pada MLA |
| :--- | :--- |
| Pertumbuhan KV cache $\mathcal{O}(n_h \cdot d_h)$ per token | Kompresi laten, KV cache jadi $\mathcal{O}(d_c)$ per token |
| Dekomposisi *low-rank* merusak sifat RoPE | **Decoupled RoPE**: rotasi dipisah ke sub-komponen khusus |
| *Trade-off* efisiensi memori vs kapasitas model | Kapasitas dijaga lewat *up-projection* yang presisi |

---

## 🏗️ Arsitektur Keseluruhan

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

### 1. Kompresi Low-Rank untuk Key & Value

$\mathbf{h}_t \in \mathbb{R}^{d_{\mathrm{model}}}$ diproyeksikan ke vektor laten bersama $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$, dengan $d_c \ll n_h \cdot d_h$:

$$
\mathbf{c}_t^{KV} = W^{DKV} \, \mathbf{h}_t
$$

*Key* dan *Value* direkonstruksi dari vektor laten via *up-projection*:

$$
\mathbf{k}_t^{C} = W^{UK} \, \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^{C} = W^{UV} \, \mathbf{c}_t^{KV}
$$

> **💡 Inti Efisiensi:** saat inferensi, hanya $\mathbf{c}_t^{KV}$ yang disimpan di KV Cache.

### 2. Kompresi Query

$$
\mathbf{c}_t^{Q} = W^{DQ} \, \mathbf{h}_t, \quad \mathbf{q}_t^{C} = W^{UQ} \, \mathbf{c}_t^{Q}
$$

### 3. Matriks Attention

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_h + d_h^R}}\right) V
$$

---

## 🔄 Decoupled RoPE

Jika RoPE diterapkan langsung ke $\mathbf{k}_t^{C}$ (hasil rekonstruksi dari laten), matriks *up-projection* $W^{UK}$ harus dieksekusi ulang secara dinamis per posisi — tidak efisien.

**Solusi:** komponen posisional dipisah. *Key Rotary* kecil dibuat langsung dari $\mathbf{h}_t$:

$$
\mathbf{k}_t^{R} = \mathrm{RoPE}\Big(W^{KR} \, \mathbf{h}_t\Big)
$$

*Key* akhir = gabungan komponen semantik laten + rotary:

$$
\mathbf{k}_t = \Big[\, \mathbf{k}_t^{C} \; \Vert \; \mathbf{k}_t^{R} \,\Big]
$$

Pola sama untuk *Query*:

$$
\mathbf{q}_t^{R} = \mathrm{RoPE}\Big(W^{QR} \, \mathbf{c}_t^{Q}\Big), \quad \mathbf{q}_t = \Big[\, \mathbf{q}_t^{C} \; \Vert \; \mathbf{q}_t^{R} \,\Big]
$$

$$
\langle f_q(x_m, m),\, f_k(x_n, n) \rangle = g(x_m, x_n, m-n)
$$

---

## 🧩 Struktur Notebook (Cell-by-Cell)

`AI_Transfomers.ipynb` berisi **32 cell**, dibagi jadi 8 bagian berurutan (harus dijalankan top-to-bottom karena setiap bagian bergantung pada variabel dari bagian sebelumnya):

| # | Bagian | Isi |
| :-- | :--- | :--- |
| 1 | **Setup Library & Dataset** | Import (`torch`, `transformers`, `datasets`), cek CUDA, login Hugging Face Hub, load dataset `nampdn-ai/tiny-strange-textbooks` |
| 2 | **Split & Tokenisasi** | Subset 50.000 sample, split train/val/test (98/1/1), tokenisasi dengan `bert-base-uncased` (`max_length=256`) |
| 3 | **Embedding & RoPE** | `nn.Embedding`, kelas `RotaryPositionEmbedding` (demo standalone) |
| 4 | **Multi-Head Latent Attention** | Kelas `MultiHeadLatentAttention` lengkap (down/up-projection, decoupled RoPE) |
| 5 | **MoE Feed Forward** | Kelas `Expert` (SwiGLU) dan `MoEFeedForward` (router top-k + shared expert + auxiliary load-balancing loss) |
| 6 | **Transformer Block** | Kelas `RMSNorm` dan `TransformerBlock` (pre-norm + residual) |
| 7 | **Training** | Kelas `Nvoin_Language_Model`, collate function, loop training dengan mixed precision, gradient accumulation, time-boxed stopping, checkpointing |
| 8 | **Inference / Prompting** | Load checkpoint, fungsi `generate()` autoregresif (temperature, top-k, top-p), sesi chat interaktif |

---

## 💻 Instalasi

Proyek ini **belum berbentuk package Python**, jadi tidak ada `pip install -r requirements.txt` dari repo. Yang perlu disiapkan:

```bash
pip install torch transformers datasets huggingface_hub matplotlib
```

Kamu juga butuh:
- **Akun Hugging Face** (notebook memanggil `login()` di cell 3 — siapkan token akses).
- **GPU disarankan.** Notebook mendeteksi CUDA otomatis, tapi tetap bisa jalan di CPU (lebih lambat).
- Waktu load dataset: menurut komentar penulis, proses ini **±26-30 menit**.

---

## ▶️ Menjalankan Notebook

Buka `AI_Transfomers.ipynb` di Jupyter/Colab dan jalankan cell **secara berurutan dari atas ke bawah** — setiap bagian bergantung pada variabel yang dibuat di bagian sebelumnya (tokenizer, dataset, model, dll didefinisikan sebagai variabel global, bukan diimpor dari module terpisah).

Ringkasan alur:
1. Login HF → load & tokenisasi dataset.
2. Definisikan kelas (`MultiHeadLatentAttention`, `MoEFeedForward`, `TransformerBlock`, `Nvoin_Language_Model`).
3. Inisialisasi model dengan hyperparameter di cell konfigurasi.
4. Jalankan loop training (checkpoint otomatis tersimpan di folder `checkpoints/`).
5. Load checkpoint dan jalankan sesi chat interaktif untuk mencoba generate teks.

> Karena semua kelas berada di cell notebook (bukan file `.py` terpisah), **belum bisa** melakukan `from model import Nvoin_Language_Model` seperti pada proyek berbasis package. Kalau kamu ingin dipakai ulang di script/aplikasi lain, kelas-kelas tersebut perlu diekstrak dulu ke file `.py` — lihat bagian [Limitasi](#limitasi--catatan-pengembangan).

---

## ⚙️ Konfigurasi Model yang Dipakai

Ini konfigurasi yang benar-benar dipakai pada cell training (bukan angka ilustratif):

```python
VOCAB_SIZE    = tokenizer.vocab_size   # bert-base-uncased
D_MODEL       = 256
N_LAYERS      = 4
N_HEADS       = 8
D_HEAD        = 32
D_ROTARY      = 16      # harus genap
D_LATENT_KV   = 64
D_LATENT_Q    = 96
D_FFN_HIDDEN  = 512
N_ROUTED_EXP  = 8
N_SHARED_EXP  = 1
TOP_K         = 2
DROPOUT       = 0.1
```

Hyperparameter training:

```python
N_EPOCHS            = 3
LEARNING_RATE        = 3e-4
WEIGHT_DECAY          = 0.01
GRAD_CLIP             = 1.0
MAX_TRAINING_HOURS    = 5      # training berhenti otomatis walau epoch belum selesai
```

Training memakai **mixed precision** (`torch.cuda.amp`) dan **gradient accumulation** untuk menghemat VRAM.

---

## 📁 Struktur Proyek (Nyata)

```text
.
├── AI_Transfomers.ipynb   # Satu-satunya file kode: dataset, model, training, inference
├── checkpoints/           # Dibuat otomatis saat training (berisi nvoin_checkpoint.pt)
└── README.md              # Dokumentasi ini
```

Tidak ada `model.py`, `mla.py`, `moe.py`, `rope.py`, `norm.py`, `block.py`, atau `train.py` terpisah di proyek ini saat ini.

---

## ⚖️ Perbandingan MLA vs MHA

| Atribut | MHA (Standar) | MLA (DeepSeek-V2) |
| :--- | :--- | :--- |
| **Penyimpanan Cache** | Vektor $K, V$ penuh per *head* | Vektor laten $\mathbf{c}_t^{KV} + \mathbf{k}_t^{R}$ |
| **Beban Memori Cache** | $\mathcal{O}(n_h \cdot d_h)$ per token | $\mathcal{O}(d_c + d_h^R)$ per token |
| **Pemrosesan Posisi** | Melekat pada vektor $K$ | Dipisahkan (*Decoupled*) |
| **Efisiensi Low-Rank** | Tidak relevan | Optimal |

---

## ⚠️ Limitasi & Catatan Pengembangan

1. **Belum ada matrix absorption saat inferensi.** $W^{UK}$ belum dilebur ke *up-projection*; notebook masih melakukan rekonstruksi penuh $K$ demi keterbacaan kode.
2. **MoE routing masih loop Python murni** — belum memakai kernel scatter/gather CUDA, jadi belum optimal untuk skala expert/batch besar.
3. **Tokenizer WordPiece (`bert-base-uncased`) dipakai untuk causal LM.** Ini bukan pilihan standar (biasanya BPE seperti GPT-2) — worth dicoba diganti kalau tujuan proyek berkembang ke arah generasi teks yang lebih natural.
4. **Struktur satu-file.** Semua kelas ada dalam satu notebook; belum ada pemisahan module (`.py`) sehingga kode belum reusable sebagai package. Kalau perlu dipakai di luar notebook, ekstrak kelas ke file terpisah (`rope.py`, `mla.py`, `moe.py`, `norm.py`, `block.py`, `model.py`) secara manual.
5. **Ada baris evaluasi redundan di akhir training** (`evaluate(nvoin_model, test_dataset, device)` dipanggil sebelum `evaluate(nvoin_model, test_loader, device)`) — baris pertama memakai `test_dataset` (bukan `DataLoader`) dan berpotensi error atau tidak diperlukan; sebaiknya dihapus.
6. **Cell demo RoPE standalone (bagian 3)** memanggil `RotaryPositionEmbedding(seq_len, embed_dim)`, padahal signature-nya `__init__(self, dim, max_seq_len)` — parameter tertukar. Tidak memengaruhi model final (MLA membuat instance RoPE-nya sendiri dengan benar), tapi menyesatkan sebagai contoh berdiri sendiri.
7. **Generasi autoregresif dasar** — KV-cache belum dioptimalkan untuk streaming; `generate()` masih memakai agregasi konteks standar.

---

## 📚 Referensi

1. DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.*
2. Su, Jianlin et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
3. Vaswani, Ashish et al. (2017). *Attention Is All You Need.*

---

## 📜 Lisensi

Belum ada file `LICENSE` di proyek ini. Tambahkan file lisensi (mis. MIT) secara eksplisit bila proyek ini ingin didistribusikan dengan lisensi tertentu.