"""
Mesin Fuzzy MADM (Fuzzy-SAW) untuk SPK Prioritas Penanganan Kesehatan Mental.

Alur metode:
    1. Fuzzifikasi   : tiap nilai kriteria -> derajat keanggotaan Rendah/Sedang/Tinggi
                       (fungsi keanggotaan segitiga via skfuzzy.trimf).
    2. Defuzzifikasi : derajat keanggotaan -> skor severity 0..1 per kriteria
                       (centroid berbobot, arah benefit/cost diperhitungkan).
    3. Agregasi SAW  : skor akhir = sum(bobot_ternormalisasi * severity).
    4. Perangkingan  : urut skor akhir menurun -> Peringkat 1 = prioritas tertinggi.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import skfuzzy as fuzz

# Pusat severity tiap himpunan fuzzy. Benefit: nilai tinggi = prioritas tinggi.
# Cost: dibalik di dalam kode (Rendah <-> Tinggi) supaya nilai rendah = prioritas tinggi.
SEVERITY_CENTER = {"Rendah": 0.15, "Sedang": 0.50, "Tinggi": 0.85}

# Ambang kategori prioritas pada skor akhir (0..100).
KATEGORI_AMBANG = [(66.0, "Tinggi"), (33.0, "Sedang"), (0.0, "Rendah")]


@dataclass(frozen=True)
class Criterion:
    """Definisi satu kriteria SPK."""

    key: str            # nama kolom pada dataframe
    label: str          # label tampilan
    direction: str      # 'benefit' (tinggi=prioritas) | 'cost' (rendah=prioritas)
    vmin: float         # batas bawah semesta
    vmax: float         # batas atas semesta
    default_weight: float
    keterangan: str = ""

    @property
    def mid(self) -> float:
        return (self.vmin + self.vmax) / 2.0

    @property
    def mf_params(self) -> dict[str, list[float]]:
        """Parameter trapmf [a, b, c, d] untuk Rendah/Sedang/Tinggi.

        Bentuk trapesium: dua himpunan ujung berbentuk bahu (shoulder) ber-plato,
        himpunan tengah trapesium. Bersinggungan (crossing) di μ=0,5.
        """
        lo, hi = self.vmin, self.vmax
        s = hi - lo
        return {
            "Rendah": [lo, lo, lo + 0.20 * s, lo + 0.40 * s],
            "Sedang": [lo + 0.20 * s, lo + 0.40 * s, lo + 0.60 * s, lo + 0.80 * s],
            "Tinggi": [lo + 0.60 * s, lo + 0.80 * s, hi, hi],
        }


@dataclass(frozen=True)
class DatasetConfig:
    """Konfigurasi dataset + kriteria yang dipakai sebagai alternatif."""

    name: str
    file: str
    id_label: str
    criteria: list[Criterion]
    label_col: str | None = None          # kolom target (referensi, bukan kriteria)
    preprocess: callable | None = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Registry dataset
# --------------------------------------------------------------------------- #
DATASETS: dict[str, DatasetConfig] = {
    "StressLevelDataset": DatasetConfig(
        name="StressLevelDataset",
        file="StressLevelDataset.csv",
        id_label="MHS",
        label_col="stress_level",
        criteria=[
            Criterion("depression", "Tingkat Depresi", "benefit", 0, 27, 0.25,
                      "Skor depresi (PHQ-style). Makin tinggi makin prioritas."),
            Criterion("anxiety_level", "Tingkat Kecemasan", "benefit", 0, 21, 0.20,
                      "Skor kecemasan (GAD-style)."),
            Criterion("study_load", "Tekanan Akademik", "benefit", 0, 5, 0.15,
                      "Beban / tekanan studi yang dirasakan."),
            Criterion("future_career_concerns", "Kekhawatiran Karier", "benefit", 0, 5, 0.10,
                      "Kekhawatiran terhadap masa depan karier."),
            Criterion("sleep_quality", "Kualitas Tidur", "cost", 0, 5, 0.15,
                      "Kualitas tidur. Makin BURUK (rendah) makin prioritas."),
            Criterion("social_support", "Dukungan Sosial", "cost", 0, 3, 0.15,
                      "Dukungan sosial. Makin RENDAH makin prioritas."),
        ],
    ),
}

# Dataset tunggal yang dipakai aplikasi.
DATASET = DATASETS["StressLevelDataset"]


# --------------------------------------------------------------------------- #
# Pemuatan data
# --------------------------------------------------------------------------- #
def load_dataset(cfg: DatasetConfig, base_dir) -> pd.DataFrame:
    df = pd.read_csv(f"{base_dir}/{cfg.file}")
    if cfg.preprocess is not None:
        df = cfg.preprocess(df)
    return df


# --------------------------------------------------------------------------- #
# Fungsi keanggotaan (untuk plot kurva)
# --------------------------------------------------------------------------- #
def membership_curves(crit: Criterion, n: int = 200):
    """Kembalikan (semesta x, dict{label: derajat}) untuk plot kurva keanggotaan."""
    x = np.linspace(crit.vmin, crit.vmax, n)
    curves = {lbl: fuzz.trapmf(x, p) for lbl, p in crit.mf_params.items()}
    return x, curves


def membership_degree(crit: Criterion, value: float) -> dict[str, float]:
    """Derajat keanggotaan satu nilai pada tiap himpunan (untuk penanda titik)."""
    return {
        lbl: float(fuzz.trapmf(np.array([value]), p)[0])
        for lbl, p in crit.mf_params.items()
    }


# --------------------------------------------------------------------------- #
# Inti perhitungan Fuzzy-SAW
# --------------------------------------------------------------------------- #
def _fuzzify_column(crit: Criterion, values: np.ndarray) -> dict[str, np.ndarray]:
    """Vektor derajat keanggotaan Rendah/Sedang/Tinggi untuk seluruh baris."""
    vals = np.clip(values.astype(float), crit.vmin, crit.vmax)
    return {lbl: fuzz.trapmf(vals, p) for lbl, p in crit.mf_params.items()}


def _defuzzify(crit: Criterion, memb: dict[str, np.ndarray]) -> np.ndarray:
    """Centroid berbobot -> severity 0..1. Arah cost membalik pusat severity."""
    centers = dict(SEVERITY_CENTER)
    if crit.direction == "cost":
        centers = {"Rendah": SEVERITY_CENTER["Tinggi"],
                   "Sedang": SEVERITY_CENTER["Sedang"],
                   "Tinggi": SEVERITY_CENTER["Rendah"]}
    num = sum(memb[l] * centers[l] for l in memb)
    den = sum(memb[l] for l in memb) + 1e-9
    return num / den


def compute(
    df: pd.DataFrame,
    cfg: DatasetConfig,
    weights: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """
    Jalankan Fuzzy-SAW penuh.

    Return dict:
        ranking      : df hasil akhir (id, kriteria mentah, skor, kategori, peringkat)
        fuzzifikasi  : derajat keanggotaan per kriteria (long-ish wide)
        defuzzifikasi: severity per kriteria + skor akhir
        bobot        : df bobot mentah & ternormalisasi
    """
    crits = cfg.criteria
    n = len(df)

    # Normalisasi bobot.
    raw_w = np.array([weights[c.key] for c in crits], dtype=float)
    if raw_w.sum() == 0:
        raw_w = np.array([c.default_weight for c in crits])
    norm_w = raw_w / raw_w.sum()
    w_map = {c.key: w for c, w in zip(crits, norm_w)}

    # ID alternatif.
    if "id" in df.columns:
        ids = df["id"].astype(str).to_numpy()
    else:
        ids = np.array([f"{cfg.id_label}-{i + 1:04d}" for i in range(n)])

    fuzz_cols: dict[str, np.ndarray] = {}
    severity_cols: dict[str, np.ndarray] = {}
    raw_cols: dict[str, np.ndarray] = {}
    skor_total = np.zeros(n)

    for c in crits:
        vals = df[c.key].to_numpy()
        raw_cols[c.label] = vals
        memb = _fuzzify_column(c, vals)
        for lbl, deg in memb.items():
            fuzz_cols[f"{c.label} · {lbl}"] = np.round(deg, 3)
        sev = _defuzzify(c, memb)
        severity_cols[c.label] = np.round(sev, 4)
        skor_total += w_map[c.key] * sev

    skor_100 = np.round(skor_total * 100, 2)
    kategori = np.array([_kategori(s) for s in skor_100])

    ranking = pd.DataFrame({"ID Mahasiswa": ids, **raw_cols})
    ranking["Skor Prioritas"] = skor_100
    ranking["Kategori"] = kategori
    ranking = ranking.sort_values("Skor Prioritas", ascending=False).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, n + 1))

    order = ranking["ID Mahasiswa"].to_numpy()  # urutan ranking

    fuzzifikasi = pd.DataFrame({"ID Mahasiswa": ids, **fuzz_cols})
    defuzzifikasi = pd.DataFrame({"ID Mahasiswa": ids, **severity_cols})
    defuzzifikasi["Skor Prioritas"] = skor_100

    # Selaraskan urutan tabel detail dengan ranking.
    fuzzifikasi = _reorder(fuzzifikasi, order)
    defuzzifikasi = _reorder(defuzzifikasi, order)

    bobot = pd.DataFrame({
        "Kriteria": [c.label for c in crits],
        "Arah": [c.direction for c in crits],
        "Bobot Input": np.round(raw_w, 3),
        "Bobot Ternormalisasi": np.round(norm_w, 4),
    })

    return {
        "ranking": ranking,
        "fuzzifikasi": fuzzifikasi,
        "defuzzifikasi": defuzzifikasi,
        "bobot": bobot,
    }


def _reorder(df: pd.DataFrame, order: np.ndarray) -> pd.DataFrame:
    df = df.set_index("ID Mahasiswa").loc[order].reset_index()
    return df


def _kategori(skor: float) -> str:
    for ambang, label in KATEGORI_AMBANG:
        if skor >= ambang:
            return label
    return "Rendah"
