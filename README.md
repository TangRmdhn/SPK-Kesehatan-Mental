# SPK Pemilihan Prioritas Penanganan Kesehatan Mental Mahasiswa

Sistem Pendukung Keputusan (SPK) untuk menentukan prioritas mahasiswa yang
memerlukan penanganan kesehatan mental, berdasarkan kriteria tingkat depresi,
kecemasan, tekanan akademik, kualitas tidur, dan faktor pendukung lainnya.

- **Metode:** Fuzzy Inference System (Mamdani) hierarkis → defuzzifikasi centroid
  → agregasi bobot antar dimensi → perangkingan.
- **Antarmuka:** Streamlit (navigasi sidebar, bobot dinamis, tombol eksekusi).
- **Mata kuliah:** Praktikum SCPK 2025/2026.
- **Kelompok:** Bintang Ramadhan (123240073) - Arsyadi Indra Hasan P. (123240062).
- **Live web:** [spk-kesehatan-mental.streamlit.app](https://spk-kesehatan-mental.streamlit.app)

## Struktur

| File | Isi |
|------|-----|
| `app.py` | Aplikasi Streamlit (3 halaman: Data, Hitung SPK, Profil Kelompok). |
| `fuzzy_engine.py` | FIS Mamdani hierarkis + basis aturan + konfigurasi kriteria. |
| `StressLevelDataset.csv` | Dataset (1.100 baris, 21 kolom). |
| `.streamlit/config.toml` | Tema antarmuka. |
| `belajar.md` | Catatan belajar: teori, rumus, contoh hitung, struktur kode. |

## Metode (ringkas)

6 kriteria dibagi **3 dimensi** (2 kriteria/dimensi):

| Dimensi | Kriteria |
|---|---|
| Psikologis | Tingkat Depresi + Tingkat Kecemasan |
| Akademik | Tekanan Akademik + Kekhawatiran Karier |
| Pendukung | Kualitas Tidur (cost) + Dukungan Sosial (cost) |

Tiap dimensi = **FIS Mamdani 9 aturan** (`IF a AND b THEN prioritas`), total **27 aturan**.
Dipakai hierarkis agar tak terjadi ledakan aturan (1 FIS 6 input = 3⁶ = 729 aturan).

## Alur perhitungan

1. **Fuzzifikasi** — nilai → derajat keanggotaan *Rendah/Sedang/Tinggi* (trapesium, `skfuzzy.trapmf`).
2. **Inferensi** — tiap aturan: `α = min(μ_a, μ_b)`.
3. **Komposisi** — agregasi MAX konsekuen yang sudah dipotong (clipping).
4. **Defuzzifikasi** — centroid → skor crisp 0–100 tiap dimensi.
5. **Agregasi bobot** — `skor = Σ(bobot_dimensi × skor_dimensi)`.
6. **Perangkingan** — urut menurun; Peringkat 1 = prioritas tertinggi.
   Kategori: Tinggi (≥66), Sedang (33–66), Rendah (<33).

## Menjalankan

Dependensi dikelola dengan **uv**.

```bash
uv sync                       # pasang dependensi dari pyproject.toml / uv.lock
uv run streamlit run app.py   # jalankan aplikasi
```

Buka `http://localhost:8501` di browser. Penjelasan lengkap ada di **`belajar.md`**.
#
