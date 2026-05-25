# Catatan Belajar — SPK Prioritas Penanganan Kesehatan Mental Mahasiswa

Dokumen ini menjelaskan **seluruh metode, teori, dan kode** yang dipakai pada proyek
akhir Praktikum SCPK. Tujuannya supaya kamu (dan anggota kelompok) paham betul tiap
bagian saat presentasi — bukan sekadar bisa menjalankan aplikasinya.

---

## Daftar Isi

1. [Gambaran Proyek](#1-gambaran-proyek)
2. [Ketentuan Tugas & Cara Dipenuhi](#2-ketentuan-tugas--cara-dipenuhi)
3. [Dataset & Kriteria](#3-dataset--kriteria)
4. [Teori Metode: SPK, MADM, dan Fuzzy](#4-teori-metode-spk-madm-dan-fuzzy)
5. [Tahapan Metode Fuzzy-SAW + Rumus](#5-tahapan-metode-fuzzy-saw--rumus)
6. [Contoh Perhitungan Manual (Step-by-Step)](#6-contoh-perhitungan-manual-step-by-step)
7. [Struktur Kode](#7-struktur-kode)
8. [Visualisasi yang Dibuat](#8-visualisasi-yang-dibuat)
9. [Setup & Cara Menjalankan](#9-setup--cara-menjalankan)
10. [Glosarium Istilah](#10-glosarium-istilah)
11. [Antisipasi Pertanyaan Presentasi](#11-antisipasi-pertanyaan-presentasi)

---

## 1. Gambaran Proyek

- **Judul:** Pemilihan Prioritas Penanganan Kesehatan Mental Mahasiswa.
- **Tujuan:** Menentukan **urutan prioritas** mahasiswa yang paling membutuhkan
  penanganan kesehatan mental, berdasarkan beberapa kriteria (tingkat depresi,
  kecemasan, tekanan akademik, kualitas tidur, dan faktor pendukung lain).
- **Metode SPK:** **Fuzzy MADM** dengan agregasi **SAW** (*Simple Additive Weighting*) —
  disebut **Fuzzy-SAW**.
- **Antarmuka:** Streamlit (web app).
- **Output:** Tabel peringkat mahasiswa, Peringkat 1 = paling prioritas.

Inti SPK: ada banyak **alternatif** (mahasiswa), dinilai pada beberapa **kriteria**,
tiap kriteria punya **bobot**. SPK menggabungkan semuanya jadi satu **skor** per
alternatif, lalu mengurutkannya.

---

## 2. Ketentuan Tugas & Cara Dipenuhi

| Ketentuan (dari PDF) | Implementasi |
|---|---|
| Output berupa GUI Streamlit | `app.py` dijalankan dengan `streamlit run` |
| Navigasi sidebar/tabs (Data, Hitung SPK, Profil) | `st.radio` di sidebar, 3 halaman |
| Tampilkan dataset mentah (`st.dataframe`) | Halaman Data → tabel interaktif |
| Input bobot dinamis (min. 3 widget) | 6 `st.slider` bobot + `st.number_input` (Top-N) + `st.selectbox` |
| Tombol eksekusi (tidak otomatis) | `st.button("Hitung Prioritas")`; hasil disimpan di `st.session_state` |
| Tabel hasil perangkingan ter-sort | Tabel di-`sort` menurun, Peringkat 1 di atas |
| Dataset CSV online, ≥250 baris | StressLevelDataset = 1.100 baris |
| Minimal 5 kriteria | Dipakai **6** kriteria |
| Fuzzy WAJIB menampilkan proses (kurva keanggotaan/tabel defuzzifikasi) | Ada tab **Kurva Keanggotaan**, **Fuzzifikasi**, **Defuzzifikasi** |
| Minimal 1 grafik | Disediakan banyak (histogram, heatmap, bar, radar, pie) |

---

## 3. Dataset & Kriteria

### Dataset

**StressLevelDataset.csv** — 1.100 baris, 21 kolom, **semua numerik** (sudah bersih,
tidak perlu encoding). Sumber: Kaggle (*Student Stress Factors*).

Alasan dipilih (dari 3 kandidat awal):
- Semua kolom numerik → langsung bisa difuzzifikasi tanpa pembersihan rumit.
- Kolomnya **persis** sesuai judul: ada depresi, kecemasan, beban studi, kualitas
  tidur, dukungan sosial.
- Memenuhi syarat (>250 baris, >5 kriteria).

> Dataset lain (Stress_Dataset & Student Depression) sempat dipertimbangkan tapi
> **tidak dipakai**: Stress_Dataset punya kolom duplikat & target teks (kotor);
> Student Depression bertipe campuran (perlu encoding). Satu dataset bersih lebih
> kuat untuk dipresentasikan.

### Kriteria (6) dan arah penilaian

Tiap kriteria punya **arah**:
- **Benefit** → nilai makin **tinggi** = prioritas penanganan makin **tinggi**.
- **Cost** → nilai makin **rendah** = prioritas penanganan makin **tinggi**.

| Kriteria | Kolom | Rentang | Arah | Bobot default | Logika |
|---|---|---|---|---|---|
| Tingkat Depresi | `depression` | 0–27 | Benefit | 0,25 | Depresi tinggi → butuh ditangani |
| Tingkat Kecemasan | `anxiety_level` | 0–21 | Benefit | 0,20 | Cemas tinggi → prioritas |
| Tekanan Akademik | `study_load` | 0–5 | Benefit | 0,15 | Beban tinggi → prioritas |
| Kekhawatiran Karier | `future_career_concerns` | 0–5 | Benefit | 0,10 | Khawatir tinggi → prioritas |
| Kualitas Tidur | `sleep_quality` | 0–5 | **Cost** | 0,15 | Tidur **buruk** → prioritas |
| Dukungan Sosial | `social_support` | 0–3 | **Cost** | 0,15 | Dukungan **rendah** → prioritas |

Total bobot default = 1,00 (tapi user bebas mengubah; nanti dinormalisasi otomatis).

---

## 4. Teori Metode: SPK, MADM, dan Fuzzy

### 4.1 SPK & MADM
**SPK** (Sistem Pendukung Keputusan) = sistem yang membantu manusia mengambil
keputusan, bukan menggantikannya. **MADM** (*Multi-Attribute Decision Making*) =
keluarga metode SPK yang memilih/merangking alternatif berdasarkan **banyak atribut
(kriteria)** sekaligus. Contoh klasik: SAW, WP, TOPSIS, AHP.

### 4.2 Kenapa Fuzzy?
Nilai kriteria itu **tidak hitam-putih**. Skor depresi 13 itu "sedang" atau "tinggi"?
Batas tegas (misal ">14 = tinggi") terlalu kaku — beda 1 angka bisa lompat kategori.
**Logika fuzzy** memodelkan ini dengan **derajat keanggotaan**: satu nilai bisa
"70% Sedang dan 30% Tinggi" sekaligus. Lebih mirip cara manusia menilai.

### 4.3 Kenapa Fuzzy-SAW, bukan Fuzzy Inference System (Mamdani/Tsukamoto)?
Ada dua jalur "Fuzzy" untuk SPK:

| Aspek | FIS (Mamdani/Tsukamoto) | **Fuzzy-SAW (dipakai)** |
|---|---|---|
| Cara kerja | Basis **aturan IF-THEN** | Fuzzifikasi → skor → bobot |
| Jumlah aturan | 6 kriteria × 3 himpunan = **243 aturan** (tak praktis ditulis tangan) | Tidak perlu aturan |
| Konsep "bobot" | Tidak ada secara eksplisit | **Ada** (cocok syarat bobot dinamis) |
| Transparansi | Sulit ditampilkan | Mudah (tabel fuzzifikasi & defuzzifikasi) |

Karena tugas **mewajibkan input bobot dinamis** dan menampilkan **proses fuzzy**,
Fuzzy-SAW paling pas: punya bobot eksplisit + mudah dipamerkan prosesnya.

### 4.4 Fungsi Keanggotaan Segitiga (Triangular MF)
Tiap kriteria dibagi 3 himpunan fuzzy: **Rendah**, **Sedang**, **Tinggi**, masing-masing
berbentuk segitiga. Segitiga didefinisikan 3 titik `[a, b, c]`: mulai naik di `a`,
puncak (μ=1) di `b`, turun lagi ke 0 di `c`.

Untuk kriteria dengan rentang `[min, max]`, dengan `mid = (min+max)/2`:
- **Rendah** = `[min, min, mid]`
- **Sedang** = `[min, mid, max]`
- **Tinggi** = `[mid, max, max]`

Rumus derajat keanggotaan segitiga (`trimf`):

```
        ⎧ 0,                 x ≤ a atau x ≥ c
μ(x) =  ⎨ (x - a)/(b - a),   a < x ≤ b
        ⎩ (c - x)/(c - b),   b < x < c
```

(Di kode dihitung otomatis oleh `skfuzzy.membership.trimf`.)

---

## 5. Tahapan Metode Fuzzy-SAW + Rumus

Empat langkah utama:

### Langkah 1 — Fuzzifikasi
Ubah tiap nilai kriteria `x` menjadi 3 derajat keanggotaan:
`μ_Rendah(x)`, `μ_Sedang(x)`, `μ_Tinggi(x)` memakai fungsi segitiga di atas.

### Langkah 2 — Defuzzifikasi (skor *severity* per kriteria)
Tiap himpunan diberi **pusat severity** (seberapa "mendesak"):

| Himpunan | Pusat (benefit) | Pusat (cost, dibalik) |
|---|---|---|
| Rendah | 0,15 | 0,85 |
| Sedang | 0,50 | 0,50 |
| Tinggi | 0,85 | 0,15 |

Untuk kriteria **cost**, pusat Rendah↔Tinggi **ditukar** — inilah cara menangani arah.
Severity dihitung dengan **centroid (rata-rata terbobot)**:

```
                μ_R·c_R + μ_S·c_S + μ_T·c_T
severity(x) =  ─────────────────────────────
                    μ_R + μ_S + μ_T
```

Hasilnya angka **0..1**: makin mendekati 1 = makin mendesak ditangani.

### Langkah 3 — Normalisasi bobot + Agregasi SAW
Bobot dari slider dinormalisasi supaya totalnya 1:

```
w_j(norm) = w_j / Σ w_k
```

Lalu skor akhir tiap mahasiswa = **jumlah berbobot** severity semua kriteria:

```
Skor = Σ ( w_j(norm) × severity_j )       → lalu ×100 (skala 0..100)
```

### Langkah 4 — Perangkingan & kategori
Urutkan skor **menurun**. Peringkat 1 = skor tertinggi = prioritas tertinggi.
Kategori: **Tinggi** (skor ≥ 66), **Sedang** (33–66), **Rendah** (< 33).

---

## 6. Contoh Perhitungan Manual (Step-by-Step)

Misal seorang mahasiswa dengan nilai:

| Kriteria | Nilai | Rentang | Arah |
|---|---|---|---|
| Depresi | 20 | 0–27 | benefit |
| Kecemasan | 15 | 0–21 | benefit |
| Tekanan Akademik | 4 | 0–5 | benefit |
| Kekhawatiran Karier | 4 | 0–5 | benefit |
| Kualitas Tidur | 1 | 0–5 | cost |
| Dukungan Sosial | 0 | 0–3 | cost |

Bobot default: 0,25 / 0,20 / 0,15 / 0,10 / 0,15 / 0,15 (total = 1, jadi normalisasi
tidak mengubah apa-apa).

**Fuzzifikasi + Defuzzifikasi tiap kriteria:**

- **Depresi = 20** (mid=13,5):
  μ_Sedang = (27−20)/(27−13,5) = 0,519; μ_Tinggi = (20−13,5)/13,5 = 0,481.
  severity = (0,519·0,5 + 0,481·0,85) / 1 = **0,669**
- **Kecemasan = 15** (mid=10,5):
  μ_Sedang = (21−15)/10,5 = 0,571; μ_Tinggi = 0,429.
  severity = 0,571·0,5 + 0,429·0,85 = **0,650**
- **Tekanan Akademik = 4** (mid=2,5):
  μ_Sedang = (5−4)/2,5 = 0,4; μ_Tinggi = 0,6.
  severity = 0,4·0,5 + 0,6·0,85 = **0,710**
- **Kekhawatiran Karier = 4** → sama dengan di atas = **0,710**
- **Kualitas Tidur = 1** (cost, mid=2,5):
  μ_Rendah = (2,5−1)/2,5 = 0,6; μ_Sedang = 1/2,5 = 0,4.
  Pusat cost: Rendah=0,85, Sedang=0,5.
  severity = 0,6·0,85 + 0,4·0,5 = **0,710**  *(tidur buruk → severity tinggi ✔)*
- **Dukungan Sosial = 0** (cost):
  μ_Rendah = 1,0 (di titik 0). Pusat cost Rendah = 0,85.
  severity = **0,850**  *(tanpa dukungan → severity tertinggi ✔)*

**Agregasi SAW:**

```
Skor = 0,25·0,669 + 0,20·0,650 + 0,15·0,710
     + 0,10·0,710 + 0,15·0,710 + 0,15·0,850
     = 0,1671 + 0,1300 + 0,1065 + 0,0710 + 0,1065 + 0,1275
     = 0,7086
```

**Skor akhir = 0,7086 × 100 ≈ 70,9 → Kategori "Tinggi".**

Artinya mahasiswa ini termasuk prioritas tinggi untuk ditangani.

---

## 7. Struktur Kode

```
Projek/
├── app.py                # Antarmuka Streamlit (3 halaman + visualisasi)
├── fuzzy_engine.py       # Mesin Fuzzy-SAW + konfigurasi dataset/kriteria
├── StressLevelDataset.csv# Dataset
├── .streamlit/config.toml# Tema warna
├── pyproject.toml        # Daftar dependensi (dikelola uv)
├── uv.lock               # Versi terkunci dependensi
├── requirements.txt      # Alternatif daftar dependensi (pip)
├── README.md             # Ringkasan + cara jalan
└── belajar.md            # (file ini)
```

### 7.1 `fuzzy_engine.py` — otak perhitungan

Berisi data + logika, **terpisah dari UI** (supaya rapi & mudah diuji).

- **`Criterion`** (dataclass): definisi satu kriteria — `key` (nama kolom), `label`,
  `direction` (benefit/cost), `vmin`, `vmax`, `default_weight`. Properti `mid` dan
  `mf_params` otomatis menghasilkan parameter segitiga `[a,b,c]` untuk Rendah/Sedang/Tinggi.
- **`DatasetConfig`** (dataclass): konfigurasi dataset — nama file, daftar kriteria,
  kolom label target.
- **`DATASETS` / `DATASET`**: registry; aplikasi memakai satu dataset (`StressLevelDataset`).
- **`SEVERITY_CENTER`**: pusat severity `{Rendah:0,15, Sedang:0,5, Tinggi:0,85}`.
- **`KATEGORI_AMBANG`**: ambang kategori (66 / 33).
- **`membership_curves()`**: hasilkan titik-titik kurva keanggotaan (untuk grafik).
- **`membership_degree()`**: derajat keanggotaan satu nilai (untuk penanda di grafik).
- **`_fuzzify_column()`**: fuzzifikasi seluruh baris satu kolom (vektor, pakai `skfuzzy.trimf`).
- **`_defuzzify()`**: centroid → severity; membalik pusat jika kriteria cost.
- **`compute()`**: fungsi inti. Menjalankan langkah 1–4 dan mengembalikan dict:
  `ranking`, `fuzzifikasi`, `defuzzifikasi`, `bobot`.

### 7.2 `app.py` — antarmuka

- **Sidebar**: navigasi (`st.radio`), info dataset, ringkasan metode.
- **Halaman Data**: kartu ringkasan, tabel mentah (`st.dataframe`), tabel definisi
  kriteria, dan tab visualisasi (distribusi, korelasi, sebaran, target).
- **Hitung SPK**:
  1. Panel **6 slider bobot** + donat bobot ternormalisasi.
  2. `number_input` Top-N + `selectbox` filter kategori + tombol **Hitung Prioritas**.
  3. Hasil disimpan ke `st.session_state` (supaya tak hitung ulang otomatis).
  4. Kartu ringkasan + **tabel ranking** (diwarnai) + tombol unduh CSV.
  5. Bar chart Top-N.
  6. Tab transparansi fuzzy: **Kurva Keanggotaan**, **Fuzzifikasi**, **Defuzzifikasi**,
     **Radar Profil**, **Sebaran Skor**.
- **Profil Kelompok**: kartu anggota + tabel info proyek.
- Fungsi pembantu: `hero()`, `metric_card()`, `section()`, `base_layout()` untuk
  menjaga tampilan konsisten.

### 7.3 `.streamlit/config.toml`
Mengatur tema (warna utama teal, font) supaya tampilan rapi & seragam.

---

## 8. Visualisasi yang Dibuat

| Grafik | Halaman | Maksud |
|---|---|---|
| Histogram per kriteria | Data | Lihat sebaran nilai tiap kriteria |
| Heatmap korelasi | Data | Hubungan antar kriteria & label target |
| Scatter 2 kriteria | Data | Pola hubungan dua kriteria (diwarnai target) |
| Bar/Pie target | Data | Komposisi label asli dataset |
| Donat bobot | Hitung SPK | Proporsi bobot ternormalisasi |
| **Kurva keanggotaan** | Hitung SPK | **Wajib** — tunjukkan fuzzifikasi + penanda nilai mahasiswa |
| Bar chart Top-N | Hitung SPK | Bandingkan skor prioritas teratas |
| Radar profil | Hitung SPK | Severity per kriteria satu mahasiswa |
| Histogram + Pie skor | Hitung SPK | Sebaran skor & komposisi kategori |

---

## 9. Setup & Cara Menjalankan

Dependensi dikelola dengan **uv** (pengganti pip/venv yang cepat).

```powershell
# Masuk folder proyek
cd "D:\Tugas\Semester 4\PrakSCPK\Projek"

# Pasang dependensi (membuat .venv otomatis dari uv.lock)
uv sync

# Jalankan aplikasi
uv run streamlit run app.py
```

Browser otomatis membuka `http://localhost:8501`. Stop dengan `Ctrl+C`.

Alternatif tanpa uv (pakai pip + requirements.txt):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Dependensi utama:** streamlit (UI), pandas/numpy (data), scikit-fuzzy (fungsi
keanggotaan), plotly (grafik interaktif), scipy/matplotlib (pendukung).

---

## 10. Glosarium Istilah

- **Alternatif** — objek yang dirangking; di sini = mahasiswa.
- **Kriteria** — aspek penilaian (depresi, tidur, dst).
- **Bobot** — tingkat kepentingan tiap kriteria.
- **Benefit/Cost** — arah kriteria (tinggi-baik vs rendah-baik bagi prioritas).
- **Himpunan fuzzy** — kategori bertingkat (Rendah/Sedang/Tinggi) dengan keanggotaan parsial.
- **Fungsi keanggotaan (μ)** — fungsi yang memberi derajat 0..1 suatu nilai pada satu himpunan.
- **Fuzzifikasi** — mengubah nilai tegas → derajat keanggotaan.
- **Defuzzifikasi** — mengubah derajat keanggotaan → satu angka tegas (di sini: severity).
- **Centroid** — metode defuzzifikasi rata-rata terbobot (titik berat).
- **Severity** — skor 0..1 seberapa mendesak satu kriteria menuntut penanganan.
- **SAW** — *Simple Additive Weighting*, penjumlahan berbobot.
- **Normalisasi bobot** — membagi tiap bobot dengan totalnya agar berjumlah 1.

---

## 11. Antisipasi Pertanyaan Presentasi

**Q: Kenapa pakai Fuzzy-SAW, bukan SAW biasa?**
SAW biasa pakai nilai mentah langsung. Fuzzy-SAW melewatkan nilai lewat fungsi
keanggotaan dulu, jadi penilaian "Rendah/Sedang/Tinggi" lebih luwes dan mirip
penilaian manusia, tidak kaku di batas angka.

**Q: Bagaimana menangani kriteria yang "kebalikan" (cost)?**
Saat defuzzifikasi, pusat severity untuk himpunan Rendah dan Tinggi **ditukar**.
Jadi untuk kualitas tidur, nilai *rendah* (tidur buruk) justru menghasilkan severity
*tinggi*.

**Q: Bobotnya dari mana?**
Bobot diatur **user** lewat slider (dinamis). Sistem hanya memberi nilai default.
Bobot dinormalisasi otomatis agar totalnya 1, sehingga skor selalu di skala 0–100.

**Q: Kenapa skor 0–100, kategori 66/33?**
Severity asalnya 0–1; dikali 100 supaya mudah dibaca sebagai "skor prioritas".
Ambang 33 & 66 membagi rentang jadi tiga sama besar (Rendah/Sedang/Tinggi).

**Q: Kenapa tidak pakai aturan IF-THEN (Mamdani)?**
Dengan 6 kriteria × 3 himpunan, butuh ratusan aturan dan tidak ada konsep bobot
eksplisit. Fuzzy-SAW lebih ringkas, punya bobot, dan prosesnya gampang ditampilkan.

**Q: Di mana letak "proses fuzzy" yang wajib ditampilkan?**
Di halaman Hitung SPK → tab **Kurva Keanggotaan** (fuzzifikasi visual),
tab **Fuzzifikasi** (tabel derajat keanggotaan), dan tab **Defuzzifikasi**
(tabel severity + skor akhir).

**Q: Apakah sistem ini mendiagnosis penyakit?**
Tidak. Ini **alat bantu prioritas** (decision *support*), bukan diagnosis medis.
Keputusan akhir tetap di tangan konselor/psikolog.
