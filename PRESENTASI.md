# Fitur Lebih Menentukan daripada Arsitektur — Autoencoder DINOv2 Ringan untuk Deteksi Anomali pada MVTec-AD

> **Judul (EN):** *Features over Architecture: A Lightweight DINOv2 Autoencoder for Unsupervised Anomaly Detection on MVTec-AD*
> **Judul slide:** *DINO-AE: Fitur Foundation Mengalahkan Transformer untuk Deteksi Anomali*
> **Subjudul:** Dari Pixel-VAE → UniAD → DINO-AE

Dokumen penjelasan lengkap untuk presentasi ke dosen. Dibaca dari atas ke bawah —
ini sebuah alur cerita, bukan kumpulan poin acak. Bagian 1–8 = konsep & alur;
Bagian 9 = penjelasan kode per-file; Bagian 10–12 = konsep kunci, hasil, kesimpulan.

---

## 1. Masalah yang diselesaikan

**Deteksi anomali industri tanpa pengawasan (unsupervised anomaly detection)** pada
dataset **MVTec-AD**. Aturannya khas dan penting:

> Model **hanya dilatih pada gambar NORMAL** (tanpa cacat). Saat pengujian, model
> harus **mendeteksi** apakah sebuah gambar cacat, dan **melokalisasi** di mana
> letak cacatnya — padahal model tidak pernah melihat satu pun contoh cacat saat
> latihan.

Kenapa begini? Karena di pabrik nyata, produk cacat itu jarang dan jenisnya
tak terhingga — tidak mungkin mengumpulkan semua contoh cacat. Jadi pendekatannya:
"pelajari yang normal, lalu apa pun yang menyimpang = anomali."

---

## 2. Dataset MVTec-AD

- **15 kategori** (10 objek seperti botol, kapsul, sekrup + 5 tekstur seperti
  karpet, kulit, ubin).
- Setiap kategori punya:
  - `train/good/` → **hanya gambar normal** (untuk latihan)
  - `test/` → campuran gambar normal + berbagai jenis cacat (untuk pengujian)
  - `ground_truth/` → **mask piksel** yang menandai letak persis setiap cacat
- **Metrik evaluasi (AUROC, 0–1, makin tinggi makin bagus):**
  - **Image-level AUROC** = seberapa baik mendeteksi gambar cacat (deteksi)
  - **Pixel-level AUROC** = seberapa tepat menunjukkan lokasi cacat (lokalisasi)

---

## 3. Ide inti yang menyatukan semuanya

Kami membangun **tiga metode**, tetapi sebenarnya itu **satu ide yang terus
disempurnakan**, dengan satu pertanyaan kunci:

> ### "Apa yang sebaiknya direkonstruksi oleh autoencoder?"

Prinsip dasar semua metode sama: **latih model membangun ulang (rekonstruksi)
data normal → saat tes, apa pun yang gagal direkonstruksi dengan baik = cacat.**
Yang berubah hanyalah *apa* yang direkonstruksi.

---

## 4. Tiga tahap

### Tahap 1 — Pixel-VAE (baseline / titik awal)
- Autoencoder yang merekonstruksi **piksel mentah** (warna gambar), satu model per
  kategori.
- **Hasil: 0.687 deteksi / 0.805 lokalisasi.**
- **Temuan:** bagus untuk objek (hazelnut 0.96), **buruk untuk tekstur**. Skor MSE
  pada karpet bahkan *terbalik* (0.31, di bawah tebakan acak) karena latent global
  tak mampu "melukis ulang" anyaman karpet. SSIM sedikit menolong.

### Tahap 2 — UniAD (mereproduksi state-of-the-art)
- Metode NeurIPS 2022. Merekonstruksi **fitur terlatih (pretrained features)**,
  bukan piksel, menggunakan **transformer**; **satu model terpadu** untuk semua 15
  kategori, plus 3 trik anti-menyontek (neighbor mask, learnable query, jitter).
- Kode resmi kami **port** agar jalan di GPU modern (RTX 4060).
- **Hasil: 0.942 / 0.960.**
- **Pelajaran:** lonjakan besar berasal dari **fitur, bukan piksel mentah**.

### Tahap 3 — DINO-AE (kontribusi kami — mengalahkan UniAD)
- Rekonstruksi **fitur foundation model DINOv2** dengan **autoencoder konvolusi
  sangat kecil** (hanya bottleneck + noise — tanpa transformer). Skor pakai
  **jarak cosine**.
- **Hasil: 0.972 / 0.966 — mengalahkan UniAD**, model jauh lebih sederhana,
  ~20 menit untuk semua 15 kategori.
- Terbaik/seri di **13 dari 15** kategori. Karpet 0.52→1.00, metal_nut 0.56→1.00.

---

## 5. Apa itu "features" (fitur)?

- **Piksel mentah** = sekadar angka warna per titik, tanpa makna. Piksel tidak
  "tahu" ia bagian goresan atau ulir sekrup.
- **Fitur** = hasil dari jaringan saraf terlatih (DINOv2). Tiap potongan (patch)
  gambar diubah jadi **vektor makna** (768 angka) yang menggambarkan *apa* yang ada
  di situ ("ini tekstur karpet normal"), bukan warnanya.

**Analogi:** piksel = cahaya mentah yang masuk ke mata; fitur = apa yang *dikenali
otak*. Atau: piksel = huruf; fitur = makna kata. Membandingkan makna jauh lebih
tahan-banting daripada membandingkan huruf demi huruf.

**Kenapa fitur menang:** membangun ulang karpet via piksel = harus melukis ulang
tiap benang (sulit). Via fitur = cukup mereproduksi "ke-normal-an karpet" (mudah).
Cacat menghasilkan fitur "tidak normal" → langsung menonjol.

---

## 6. Alur pipeline DINO-AE: KONTROL vs DATA

Penting dibedakan dua jenis "alur":

### Alur KONTROL (file mana yang menjalankan)
`train_dino_ae.py` adalah **entry point / "sutradara"** — file yang kamu jalankan.
Dia **memanggil** semua file lain (bukan langkah terakhir):
```
train_dino_ae.py  ← dijalankan (sutradara)
   ├─ dataset.py        (muat gambar)
   ├─ dino_features.py  (piksel → fitur)
   ├─ feat_recon.py     (rekonstruksi fitur)
   └─ common.py         (skor)
```

### Alur DATA (urutan diproses)
Di dalam `train_dino_ae.py`, datanya mengalir:
```
dataset.py → dino_features.py → feat_recon.py → jarak cosine → skor + peta → results.csv
 (gambar)     (jadi fitur)       (rekonstruksi)
```

### Dua fase
**Latihan** (hanya gambar normal): muat gambar → DINOv2 (beku) jadi fitur, di-cache
→ latih AE kecil (encode→bottleneck→decode) dengan loss cosine → AE jadi ahli
membangun ulang fitur normal.

**Pengujian:** gambar tes → DINOv2 → fitur → AE rekonstruksi → jarak cosine per
patch = peta anomali → skor gambar (1% terburuk) + lokalisasi (vs mask).

**Kenapa cacat tertangkap:** AE hanya belajar fitur normal. Cacat = fitur yang
belum pernah dilihat → direkonstruksi salah → jarak cosine tinggi → terdeteksi.

---

## 7. Mengapa DINO-AE jauh lebih cepat dari UniAD

| | UniAD | DINO-AE |
|---|---|---|
| backbone (bagian berat & beku) | dijalankan **tiap epoch** (×100–1000) | **sekali saja, lalu di-cache** |
| yang dilatih | transformer 8-layer (attention O(N²)) | AE konvolusi kecil |
| epoch | 100 (kami) / 1000 (paper) | ~200 (model mungil) |
| total waktu | ~2 jam | ~20 menit |

**Inti:** DINO-AE **memisahkan & meng-cache** bagian beku yang berat (DINOv2
dihitung sekali), sedangkan UniAD **menghitung ulang backbone beku tiap epoch**
sambil melatih transformer besar. Bahkan ~20 menit DINO-AE itu mayoritasnya
ekstraksi fitur sekali; latihan AE-nya hitungan detik.

---

## 8. Konsep kunci

### 8.1 AUROC — apa & kenapa dipakai
**AUROC = Area Under the ROC Curve.** Model mengeluarkan **skor** kontinu (bukan
ya/tidak); AUROC mengukur seberapa baik skor itu **memisahkan normal dari cacat di
semua ambang sekaligus**.

> Interpretasi: AUROC = peluang sebuah gambar **cacat** mendapat skor **lebih
> tinggi** daripada gambar **normal**. 1.0 = sempurna; 0.5 = tebak koin; <0.5 =
> terbalik (seperti MSE karpet 0.31).

Kenapa AUROC, bukan accuracy/F1:
1. **Bebas-ambang** — tak perlu memilih cutoff arbitrer (deteksi anomali = skor).
2. **Tahan kelas timpang** — accuracy menyesatkan saat data tak seimbang.
3. **Standar MVTec** — semua paper (UniAD, PatchCore) pakai AUROC → bisa
   dibandingkan langsung dengan angka kami.

### 8.2 Skor cosine & hubungannya dengan SSIM
Tiap patch = vektor 768 angka. **Cosine** mengukur **sudut antar vektor**,
mengabaikan panjangnya: `1 − (A·B)/(|A||B|)`. Makna fitur ada di **arah**, bukan
besar — jadi cosine membandingkan makna, tahan terhadap perbedaan skala.

Hubungan dengan SSIM: **bukan rumus sama, tapi peran sama** —

| | Domain | Skor naif | Skor "sadar-makna" |
|---|---|---|---|
| piksel | gambar | MSE | **SSIM** |
| fitur | vektor | L2 (UniAD) | **Cosine** (DINO-AE) |

Jadi **cosine = "SSIM versi ruang-fitur"**: sama-sama membandingkan pola/makna,
bukan nilai mentah. (Analog dalam peran, bukan turunan matematis.)

### 8.3 Konstanta normalisasi ImageNet
`_IMAGENET_MEAN`/`_IMAGENET_STD` = rata-rata & simpangan baku kanal R/G/B dari
dataset ImageNet. Dipakai untuk menormalisasi gambar (`(x−mean)/std`) **persis
seperti saat DINOv2 dilatih** — kalau tidak, fiturnya jadi kacau.

---

## 9. Penjelasan kode per-file

### `dataset.py` — pemuat data
Mewarisi `Dataset` PyTorch. Pengambilan data terjadi **dua tahap**:
1. **`__init__` (mendaftar path):** menyiapkan transformasi (Resize, Grayscale
   opsional, `ToTensor`→[0,1]) dan membangun daftar `samples`. Untuk `split=train`
   → hanya `train/good` (label 0). Untuk `split=test` → telusuri semua subfolder,
   nama folder menentukan label (`good`→0, lainnya→1) + cari mask di `ground_truth`.
   **Belum membuka file gambar di sini.**
2. **`__getitem__` (memuat saat diminta / lazy):** baru membuka 1 gambar, terapkan
   transformasi. `train` → kembalikan gambar saja; `test` → `(gambar, label, mask)`.
   Mask di-resize NEAREST + dibinerkan (`>0.5`) agar tetap 0/1.

> Loader inilah yang **mewujudkan aturan unsupervised**: saat latihan, mustahil
> model melihat cacat karena hanya menarik dari `train/good`.

### `dino_features.py` — pengubah piksel → fitur
Kelas `DinoV2Features`:
- **`__init__`:** muat DINOv2 (`torch.hub`), **bekukan** (`eval()` +
  `requires_grad=False`) → tidak dilatih, hanya dipakai. `img_size` wajib kelipatan
  14 (patch DINOv2 = 14px → 224/14 = grid 16×16). `embed_dim=768`. Simpan buffer
  mean/std ImageNet.
- **`forward(x)` (`@torch.no_grad`):** normalisasi ImageNet → `forward_features` →
  ambil `x_norm_patchtokens` `(B,256,768)` → ubah jadi peta spasial `(B,768,16,16)`.

> Output: peta fitur (bukan gambar). DINOv2 dibekukan = kita pinjam "mata pintar"
> yang sudah dilatih jutaan gambar.

### `feat_recon.py` — autoencoder kecil (satu-satunya yang dilatih)
Kelas `FeatRecon`. Tujuan: rekonstruksi fitur **normal** dengan baik, tapi
**tak bisa menyalin cacat** (identity shortcut). Dua pertahanan:
- **Encoder** (3 konvolusi): mampatkan `16×16×768` → **`8×8×128`** (~24× kompresi).
  Model dipaksa **meringkas**, bukan menghafal.
- **Decoder** (transpose-conv): kembalikan `8×8×128` → `16×16×768`.
- **`add_noise`:** saat latihan, tambahkan noise berskala-norm (jitter) → objektif
  *denoising* (saat tes tidak ada noise).
- **`forward`:** `add_noise → encoder → decoder`. Loss dibandingkan dengan fitur
  **bersih** → "diberi fitur ber-noise, kembalikan versi normalnya."

> Kompresi + konteks spasial + noise = pengganti **sederhana** untuk 3 trik
> rumit UniAD.

### `train_dino_ae.py` — sutradara + algoritma
Bukan sekadar pembungkus; berisi logika inti:
- **`cosine_distance_map`** — rumus skor anomali (`1−cosine` per patch).
- **`extract_features`** — jalankan DINOv2 ke semua gambar, **cache** fiturnya
  (sekali saja → sumber kecepatan).
- **`train_ae`** — LOOP pelatihan: optimizer Adam, epoch, forward, cosine loss,
  backprop.
- **`evaluate`** — peta anomali → skor gambar (`_image_score`) + peta piksel →
  **AUROC** (sklearn).
- **`run_category`** — rangkai 1 kategori (dataset → ekstrak → latih → evaluasi →
  tulis CSV).
- **`main`** — baca hyperparameter (argparse), muat DINOv2 sekali, loop 15
  kategori, tulis `results.csv`.

### `common.py` — utilitas bersama
- **`ALL_CATEGORIES`** — daftar 15 kategori (dipakai kedua pipeline).
- **`_image_score`** — ubah peta anomali jadi 1 skor: rata-rata **1% piksel
  terburuk** (bukan seluruh gambar), supaya cacat kecil tak "tenggelam". Setara
  max-pooling UniAD. Dipakai bersama VAE & DINO-AE → penilaian konsisten/adil.

---

## 10. Tesis utama

Tiga tahap mengisolasi **satu variabel** tiap langkah:
- **VAE → UniAD:** piksel mentah → **fitur** (lonjakan terbesar).
- **UniAD → DINO-AE:** fitur EfficientNet → **fitur foundation DINOv2**, dan
  transformer+trik → **AE kecil + bottleneck**.

> **Kesimpulan:** faktor dominan **bukan** kecanggihan arsitektur, melainkan
> **kualitas fitur** yang direkonstruksi. Fitur foundation kuat + AE denoising kecil
> sudah cukup **mengalahkan** transformer SOTA yang dirancang khusus.

---

## 11. Hasil & perbandingan

| Metode | Deteksi (AUROC) | Lokalisasi (AUROC) | Model | Waktu latih |
|---|---|---|---|---|
| Pixel-VAE (baseline) | 0.687 | 0.805 | conv VAE ×15 | ~5 jam |
| UniAD (SOTA, direproduksi) | 0.942 | 0.960 | transformer + 3 trik | ~2 jam |
| **DINO-AE (kami)** | **0.972** | **0.966** | **AE konvolusi kecil ×15** | **~20 menit** |

Sorotan per kategori (deteksi image-level):

| Kategori | VAE | UniAD | DINO-AE |
|---|---|---|---|
| karpet | 0.515 | 0.997 | **1.000** |
| metal_nut | 0.555 | 0.934 | **1.000** |
| ubin (tile) | 0.855 | 0.991 | **1.000** |
| kapsul | 0.773 | 0.807 | **0.952** |
| sekrup | 0.438 | 0.845 | **0.887** |

---

## 12. Kesimpulan & catatan jujur

**Kesimpulan:** Kami membangun tiga metode deteksi anomali berbasis rekonstruksi;
metode buatan sendiri (DINO-AE) mengungguli baseline SOTA (UniAD) dengan model yang
jauh lebih sederhana dan cepat — membuktikan bahwa **kualitas fitur yang dominan**.

**Catatan jujur (untuk antisipasi pertanyaan dosen):**
- UniAD kami latih 100 epoch (paper 1000 → ~0.967); tetap perbandingan adil.
- VAE = 15 model spesialis; UniAD = 1 model terpadu (setting lebih sulit).
- DINO-AE memakai backbone foundation **beku** — kebaruannya pada **kombinasi +
  pembuktian** bahwa fitur yang menentukan, bukan backbone baru.
- Cosine vs SSIM = analog dalam **peran**, bukan rumus identik.

---

## 13. Struktur kode (file)

| File | Fungsi |
|---|---|
| `train_dino_ae.py` | **entry point DINO-AE** (sutradara + algoritma) |
| `dino_features.py` | ekstraktor DINOv2 beku (piksel → fitur) |
| `feat_recon.py` | autoencoder kecil (yang dilatih) |
| `dataset.py` | pemuat gambar MVTec (dipakai bersama) |
| `common.py` | utilitas bersama (daftar kategori, skor top-k) |
| `visualize_features.py` | visualisasi fitur DINOv2 (PCA → RGB) |
| `baseline_vae/` | metode baseline VAE |
| `UniAD/` | repo UniAD (lihat `UNIAD_SETUP.md`) |

**Cara menjalankan (dari folder root):**
```
python train_dino_ae.py --epochs 200 --save-model        # DINO-AE, semua 15
python baseline_vae/main.py --img-size 256 --epochs 100   # VAE baseline
```

*Dilatih pada laptop dengan GPU RTX 4060 (8 GB).*
