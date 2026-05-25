# Catatan Belajar — SPK Prioritas Penanganan Kesehatan Mental Mahasiswa

Dokumen ini menjelaskan **seluruh metode, teori, dan kode** proyek akhir Praktikum
SCPK. Metode yang dipakai: **Fuzzy Inference System (FIS) Mamdani hierarkis**.
Tujuannya supaya paham betul tiap bagian saat presentasi.

---

## Daftar Isi

1. [Gambaran Proyek](#1-gambaran-proyek)
2. [Ketentuan Tugas & Cara Dipenuhi](#2-ketentuan-tugas--cara-dipenuhi)
3. [Dataset & Kriteria](#3-dataset--kriteria)
4. [Teori: Fuzzy Logic & FIS Mamdani](#4-teori-fuzzy-logic--fis-mamdani)
5. [Tahapan Metode + Rumus](#5-tahapan-metode--rumus)
6. [Contoh Perhitungan (Step-by-Step)](#6-contoh-perhitungan-step-by-step)
7. [Struktur Kode](#7-struktur-kode)
8. [Visualisasi](#8-visualisasi)
9. [Setup & Cara Menjalankan](#9-setup--cara-menjalankan)
10. [Glosarium](#10-glosarium)
11. [Antisipasi Pertanyaan Presentasi](#11-antisipasi-pertanyaan-presentasi)

---

## 1. Gambaran Proyek

- **Judul:** Pemilihan Prioritas Penanganan Kesehatan Mental Mahasiswa.
- **Tujuan:** Menentukan **urutan prioritas** mahasiswa yang paling membutuhkan
  penanganan kesehatan mental.
- **Metode SPK:** **Fuzzy Inference System (Mamdani)** yang disusun **hierarkis**
  (3 dimensi), lalu output tiap dimensi digabung dengan **bobot** antar dimensi.
- **Antarmuka:** Streamlit. **Output:** tabel peringkat, Peringkat 1 = paling prioritas.

Inti: tiap mahasiswa (alternatif) dinilai pada 6 kriteria. Kriteria diproses lewat
**aturan IF-THEN** (basis pengetahuan), menghasilkan skor prioritas 0–100, lalu diurutkan.

---

## 2. Ketentuan Tugas & Cara Dipenuhi

| Ketentuan (PDF) | Implementasi |
|---|---|
| GUI Streamlit | `app.py`, dijalankan `streamlit run` |
| Navigasi (Data, Hitung SPK, Profil) | `st.radio` sidebar, 3 halaman |
| Tampilkan dataset (`st.dataframe`) | Halaman Data → tabel interaktif |
| Input bobot dinamis (min. 3 widget) | **3 `st.slider`** bobot dimensi + `st.number_input` (Top-N) + `st.selectbox` |
| Tombol eksekusi (tidak otomatis) | `st.button("Hitung Prioritas")` + `st.session_state` |
| Tabel hasil ter-sort | urut menurun, Peringkat 1 di atas |
| Dataset ≥250 baris, ≥5 kriteria | 1.100 baris, **6** kriteria |
| **Fuzzy WAJIB tampilkan proses** | basis aturan (27), kurva keanggotaan (input + output), tabel fuzzifikasi, inspeksi firing aturan, tabel defuzzifikasi |
| Min. 1 grafik | banyak (histogram, heatmap, kurva, bar, pie) |

---

## 3. Dataset & Kriteria

**StressLevelDataset.csv** — 1.100 baris, 21 kolom numerik (Kaggle, *Student Stress
Factors*). Dipilih karena bersih (tanpa encoding) dan kolomnya sesuai judul.

6 kriteria dipakai, dikelompokkan jadi **3 dimensi** (tiap dimensi 2 kriteria):

| Dimensi | Kriteria | Kolom | Rentang | Arah |
|---|---|---|---|---|
| **Psikologis** | Tingkat Depresi | `depression` | 0–27 | benefit |
|  | Tingkat Kecemasan | `anxiety_level` | 0–21 | benefit |
| **Akademik** | Tekanan Akademik | `study_load` | 0–5 | benefit |
|  | Kekhawatiran Karier | `future_career_concerns` | 0–5 | benefit |
| **Pendukung** | Kualitas Tidur | `sleep_quality` | 0–5 | **cost** |
|  | Dukungan Sosial | `social_support` | 0–3 | **cost** |

- **Benefit**: nilai tinggi → prioritas tinggi.
- **Cost**: nilai rendah → prioritas tinggi (mis. tidur buruk, dukungan minim).

---

## 4. Teori: Fuzzy Logic & FIS Mamdani

### 4.1 Logika fuzzy
Nilai tidak hitam-putih. Skor depresi 11 itu "sedang" atau "tinggi"? Batas tegas terlalu
kaku. Fuzzy memakai **derajat keanggotaan** 0–1: satu nilai bisa "60% Sedang, 40% Tinggi".

### 4.2 Fungsi keanggotaan (trapesium)
Tiap kriteria dibagi 3 himpunan: **Rendah/Sedang/Tinggi**, bentuk **trapesium** 4 titik
`[a,b,c,d]` (naik a→b, plato b→c, turun c→d). Titik potong di 20/40/60/80% rentang →
himpunan bertetangga bersinggungan di μ=0,5. Dihitung `skfuzzy.trapmf`.

### 4.3 Fuzzy Inference System (FIS)
FIS = sistem yang mengambil keputusan dari **basis aturan IF-THEN**. Empat tahap:
**fuzzifikasi → inferensi (aturan) → komposisi → defuzzifikasi**. Tiga varian umum:

| Varian | Konsekuen aturan | Defuzzifikasi |
|---|---|---|
| **Mamdani** (dipakai) | himpunan fuzzy | centroid (center of gravity) |
| Sugeno | fungsi linier/konstanta | rata-rata terbobot |
| Tsukamoto | himpunan monoton | rata-rata terbobot |

Dipakai **Mamdani** karena paling intuitif & paling lazim diajarkan; konsekuen berupa
himpunan "prioritas Rendah/Sedang/Tinggi" yang gampang divisualkan.

### 4.4 Masalah ledakan aturan → solusi hierarkis
Bila 1 FIS memuat 6 input × 3 himpunan → **3⁶ = 729 aturan**. Tidak praktis ditulis.
**Solusi:** susun **hierarkis**. 6 kriteria dibagi 3 dimensi (2 kriteria/dimensi). Tiap
dimensi cukup **3×3 = 9 aturan**. Total **27 aturan** (bisa ditulis lengkap). Output
ketiga dimensi (skor 0–100) digabung dengan **bobot** (lapisan SAW antar dimensi) —
sekaligus memenuhi syarat bobot dinamis.

> Catatan: aturan menangkap **interaksi dalam satu dimensi** (mis. depresi + kecemasan);
> bobot menangkap **kepentingan antar dimensi**. Kurva keanggotaan **tidak** dipengaruhi
> bobot — itu memang benar secara teori, karena bobot baru main di tahap agregasi.

---

## 5. Tahapan Metode + Rumus

### Langkah 1 — Fuzzifikasi
Tiap nilai → `μ_Rendah, μ_Sedang, μ_Tinggi` lewat trapesium.

### Langkah 2 — Basis aturan (per dimensi, 9 aturan)
Aturan dibangkitkan sistematis & sadar-arah. Beri ordinal "dorongan ke prioritas":
- benefit: `Rendah=0, Sedang=1, Tinggi=2`
- cost   : `Rendah=2, Sedang=1, Tinggi=0`  *(dibalik)*

Untuk pasangan term `(a, b)`, jumlahkan `s = ord(a) + ord(b)` (0..4):
`s ≥ 3 → Tinggi`, `s = 2 → Sedang`, `s ≤ 1 → Rendah`.

Contoh dimensi **Pendukung** (Kualitas Tidur *cost* + Dukungan Sosial *cost*):
```
IF Tidur=Rendah AND Dukungan=Rendah THEN Prioritas=Tinggi   (tidur buruk + tanpa dukungan)
IF Tidur=Tinggi AND Dukungan=Tinggi THEN Prioritas=Rendah   (tidur baik + dukungan kuat)
```

### Langkah 3 — Inferensi (firing strength)
Operator AND = **MIN**:
```
α_aturan = min( μ_a , μ_b )
```

### Langkah 4 — Komposisi
Tiap konsekuen dipotong (clipping) setinggi `α`, lalu seluruh aturan **diagregasi MAX**:
```
μ_keluaran(z) = max_aturan [ min( α_aturan , μ_konsekuen(z) ) ]
```

### Langkah 5 — Defuzzifikasi (centroid)
Ubah himpunan keluaran jadi skor crisp dengan **center of gravity**:
```
            Σ z · μ_keluaran(z)
skor_dim =  ───────────────────     (z = 0..100)
              Σ μ_keluaran(z)
```
Referensi: bila hanya **Tinggi** aktif penuh → skor ≈ **84,7**; hanya **Sedang** → **50**;
hanya **Rendah** → **15,3**. Firing parsial menghasilkan nilai di antaranya.

### Langkah 6 — Agregasi bobot + perangkingan
```
Skor = Σ ( bobot_dimensi(ternormalisasi) × skor_dimensi )
```
Urut menurun. Kategori: **Tinggi** ≥66, **Sedang** 33–66, **Rendah** <33.

---

## 6. Contoh Perhitungan (Step-by-Step)

Mahasiswa **MHS-0340** (peringkat 1, data nyata):

**Fuzzifikasi:**

| Kriteria | Nilai | μ Rendah | μ Sedang | μ Tinggi |
|---|---|---|---|---|
| Depresi | 14 | 0 | **1,0** | 0 |
| Kecemasan | 17 | 0 | 0 | **1,0** |
| Tekanan Akademik | 5 | 0 | 0 | **1,0** |
| Kekhawatiran Karier | 4 | 0 | 0 | **1,0** |
| Kualitas Tidur (cost) | 2 | 0 | **1,0** | 0 |
| Dukungan Sosial (cost) | 0 | **1,0** | 0 | 0 |

**Inferensi + Defuzzifikasi tiap dimensi:**

- **Psikologis**: aturan aktif `IF Depresi=Sedang AND Kecemasan=Tinggi THEN Tinggi`
  (α=min(1;1)=1). Hanya Tinggi aktif → centroid = **84,7**.
- **Akademik**: `IF Tek.Akademik=Tinggi AND Karier=Tinggi THEN Tinggi` (α=1) → **84,7**.
- **Pendukung**: `IF Tidur=Sedang AND Dukungan=Rendah THEN Tinggi` (α=1) → **84,7**.
  *(tidur sedang + tanpa dukungan = mendesak ✔)*

**Agregasi (bobot default 0,45 / 0,30 / 0,25):**
```
Skor = 0,45·84,7 + 0,30·84,7 + 0,25·84,7 = 84,7  → Kategori "Tinggi"
```

> **Catatan firing parsial:** kalau suatu nilai jatuh di *lereng* trapesium (bukan plato),
> dua himpunan aktif sekaligus → **dua aturan menyala** dengan α < 1, dan centroid keluaran
> jatuh di antara dua pusat kategori. Di sinilah fuzzy "menghaluskan" keputusan, beda dari
> aturan tegas.

---

## 7. Struktur Kode

```
Projek/
├── app.py                 # Antarmuka Streamlit (3 halaman + visualisasi)
├── fuzzy_engine.py        # FIS Mamdani hierarkis + aturan + kriteria
├── StressLevelDataset.csv # Dataset
├── .streamlit/config.toml # Tema
├── pyproject.toml         # Dependensi (uv)
├── uv.lock / requirements.txt
├── README.md
└── belajar.md             # (file ini)
```

### 7.1 `fuzzy_engine.py`

- **`Criterion`** — definisi kriteria (`key`, `label`, `direction`, `vmin`, `vmax`).
  Properti `mf_params` → parameter trapesium Rendah/Sedang/Tinggi.
- **`Group`** — satu dimensi (2 `Criterion`). Properti `rules` → 9 aturan via `_gen_rules`.
- **`_gen_rules(ca, cb)`** — bangkitkan 9 aturan IF-THEN sadar-arah (ordinal benefit/cost).
- **`GROUPS`** — daftar 3 dimensi (Psikologis/Akademik/Pendukung) + bobot default.
- **`OUT_MF` / `output_curves()`** — fungsi keanggotaan keluaran prioritas (0–100).
- **`membership_curves` / `membership_degree`** — kurva & derajat keanggotaan (untuk plot).
- **`_group_inference(group, df)`** — Mamdani vektorisasi: fuzzifikasi → `α=min` →
  komposisi MAX → centroid. Kembalikan skor + derajat keanggotaan.
- **`rule_firing(group, values)`** — tabel α tiap aturan untuk **satu** mahasiswa (inspeksi).
- **`rule_base_table()`** — seluruh 27 aturan untuk ditampilkan.
- **`compute(df, weights)`** — jalankan 3 FIS, agregasi bobot, ranking, kategori. Kembalikan
  `ranking`, `fuzzifikasi`, `defuzzifikasi`, `bobot`.

### 7.2 `app.py`
3 halaman. Halaman **Hitung SPK**: slider bobot dimensi + donat, tabel basis aturan,
tombol eksekusi, tabel ranking, dan tab transparansi (Kurva Keanggotaan input/output,
Fuzzifikasi, **Inferensi Aturan** (firing per mahasiswa), Defuzzifikasi, Sebaran Skor).

---

## 8. Visualisasi

| Grafik/Tabel | Letak | Maksud |
|---|---|---|
| Histogram, heatmap, scatter, komposisi target | Data | Eksplorasi distribusi |
| **Kurva keanggotaan** (input + keluaran) | Data & SPK | Tunjukkan fuzzifikasi |
| Donat bobot dimensi | SPK | Proporsi bobot |
| **Tabel basis aturan (27)** | SPK | Inti FIS |
| **Inspeksi firing aturan** per mahasiswa | SPK | Aturan mana aktif + α |
| Tabel fuzzifikasi & defuzzifikasi | SPK | Derajat keanggotaan & skor dimensi |
| Bar Top-N, histogram + pie skor | SPK | Hasil & sebaran |

---

## 9. Setup & Cara Menjalankan

```bash
cd "D:\Tugas\Semester 4\PrakSCPK\Projek"
uv sync                       # pasang dependensi (buat .venv)
uv run streamlit run app.py   # jalankan
```
Buka `http://localhost:8501`. Stop: `Ctrl+C`.

Tanpa uv: `python -m venv .venv` → aktifkan → `pip install -r requirements.txt` →
`streamlit run app.py`.

---

## 10. Glosarium

- **Alternatif** — yang dirangking (mahasiswa). **Kriteria** — aspek penilaian.
- **Himpunan fuzzy** — kategori bertingkat (Rendah/Sedang/Tinggi) dengan keanggotaan parsial.
- **Fungsi keanggotaan (μ)** — pemetaan nilai → derajat 0..1.
- **Fuzzifikasi** — nilai tegas → derajat keanggotaan.
- **FIS** — *Fuzzy Inference System*, sistem berbasis aturan IF-THEN.
- **Aturan / Rule** — `IF antiseden AND … THEN konsekuen`.
- **Antiseden** — bagian "IF". **Konsekuen** — bagian "THEN".
- **α-predikat (firing strength)** — kekuatan aturan = `min(μ antiseden)`.
- **Komposisi** — agregasi keluaran semua aturan (di sini MAX).
- **Defuzzifikasi** — himpunan keluaran → skor tegas. **Centroid** — titik berat.
- **Mamdani** — varian FIS dengan konsekuen himpunan fuzzy + defuzzifikasi centroid.
- **Benefit/Cost** — arah kriteria. **Bobot** — kepentingan antar dimensi.

---

## 11. Antisipasi Pertanyaan Presentasi

**Q: Metode apa yang dipakai?**
Fuzzy Inference System **Mamdani**, disusun **hierarkis** (3 dimensi), output digabung
dengan bobot antar dimensi.

**Q: Kenapa hierarkis, bukan satu FIS besar?**
Satu FIS untuk 6 input = 3⁶ = 729 aturan, tak mungkin ditulis. Hierarkis memecahnya jadi
3 FIS kecil (9 aturan tiap-tiap) = 27 aturan total, tetap menangkap interaksi antar kriteria.

**Q: Bagaimana aturan dibuat?**
Per dimensi 3×3 = 9 kombinasi term. Konsekuen ditentukan dari jumlah "dorongan prioritas"
kedua antiseden (sadar arah benefit/cost). Lengkapnya ada di tab **Basis Aturan**.

**Q: Bagaimana menangani kriteria cost (kebalikan)?**
Di aturan: mis. `IF KualitasTidur=Tinggi (baik) THEN Prioritas=Rendah`. Arah ditangani
langsung di basis aturan, bukan di kurva.

**Q: Operator AND-nya apa? Defuzzifikasinya apa?**
AND = **MIN** (firing strength). Komposisi = **MAX**. Defuzzifikasi = **centroid**
(center of gravity) pada semesta 0–100.

**Q: Di mana bobot berperan? Kenapa kurva tak ikut bobot?**
Bobot dipakai di **agregasi antar dimensi** (`Σ bobot × skor_dimensi`), bukan di kurva.
Kurva hanya untuk fuzzifikasi nilai → derajat; ini sesuai teori FIS.

**Q: Di mana "proses fuzzy" yang wajib ditampilkan?**
Tab **Basis Aturan** (27 aturan), **Kurva Keanggotaan** (input & keluaran), **Fuzzifikasi**,
**Inferensi Aturan** (aturan aktif + α tiap mahasiswa), dan **Defuzzifikasi** (skor dimensi).

**Q: Apakah ini mendiagnosis penyakit?**
Tidak. Ini **alat bantu prioritas**, bukan diagnosis. Keputusan akhir tetap di konselor/psikolog.
