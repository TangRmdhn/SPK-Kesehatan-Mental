"""
Mesin Fuzzy Inference System (FIS) Mamdani hierarkis untuk SPK Prioritas
Penanganan Kesehatan Mental Mahasiswa.

Kenapa hierarkis? 6 kriteria x 3 himpunan = 729 aturan bila satu FIS. Tidak praktis.
Solusi: 6 kriteria dikelompokkan jadi 3 dimensi (masing-masing 2 kriteria). Tiap
dimensi punya FIS Mamdani kecil (3x3 = 9 aturan) yang menangkap interaksi antar dua
kriteria. Output ketiga dimensi (skor 0-100) digabung dengan bobot dinamis (SAW antar
dimensi) lalu dirangking.

Alur tiap FIS dimensi (Mamdani):
    1. Fuzzifikasi  : nilai -> derajat keanggotaan Rendah/Sedang/Tinggi (trapmf).
    2. Inferensi    : tiap aturan IF a AND b THEN prioritas; alpha = min(mu_a, mu_b).
    3. Komposisi    : agregasi MAX dari konsekuen yang sudah dipotong (clipping).
    4. Defuzzifikasi: centroid (center of gravity) -> skor crisp 0-100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import skfuzzy as fuzz

TERMS = ["Rendah", "Sedang", "Tinggi"]

# Ordinal "dorongan ke prioritas" tiap himpunan, tergantung arah kriteria.
ORD_BENEFIT = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}   # nilai tinggi -> prioritas tinggi
ORD_COST = {"Rendah": 2, "Sedang": 1, "Tinggi": 0}      # nilai rendah -> prioritas tinggi

KATEGORI_AMBANG = [(66.0, "Tinggi"), (33.0, "Sedang"), (0.0, "Rendah")]

# Semesta keluaran prioritas (0-100) + fungsi keanggotaan keluaran.
Z = np.linspace(0, 100, 101)
OUT_PARAMS = {
    "Rendah": [0, 0, 20, 40],
    "Sedang": [25, 40, 60, 75],
    "Tinggi": [60, 80, 100, 100],
}
OUT_MF = {t: fuzz.trapmf(Z, p) for t, p in OUT_PARAMS.items()}


@dataclass(frozen=True)
class Criterion:
    """Definisi satu kriteria (variabel masukan FIS)."""

    key: str
    label: str
    direction: str      # 'benefit' (tinggi=prioritas) | 'cost' (rendah=prioritas)
    vmin: float
    vmax: float
    keterangan: str = ""

    @property
    def mf_params(self) -> dict[str, list[float]]:
        """Parameter trapmf [a,b,c,d] untuk Rendah/Sedang/Tinggi (partisi 20/40/60/80%)."""
        lo, hi = self.vmin, self.vmax
        s = hi - lo
        return {
            "Rendah": [lo, lo, lo + 0.20 * s, lo + 0.40 * s],
            "Sedang": [lo + 0.20 * s, lo + 0.40 * s, lo + 0.60 * s, lo + 0.80 * s],
            "Tinggi": [lo + 0.60 * s, lo + 0.80 * s, hi, hi],
        }


def _gen_rules(ca: Criterion, cb: Criterion) -> list[tuple[str, str, str]]:
    """Bangkitkan 9 aturan IF a AND b THEN prioritas, sadar arah benefit/cost.

    Logika: jumlahkan ordinal dorongan kedua antiseden (0..4). s>=3 -> Tinggi,
    s==2 -> Sedang, s<=1 -> Rendah. Konsisten & mudah dijelaskan.
    """
    oa = ORD_COST if ca.direction == "cost" else ORD_BENEFIT
    ob = ORD_COST if cb.direction == "cost" else ORD_BENEFIT
    rules = []
    for ta in TERMS:
        for tb in TERMS:
            s = oa[ta] + ob[tb]
            tout = "Tinggi" if s >= 3 else ("Sedang" if s == 2 else "Rendah")
            rules.append((ta, tb, tout))
    return rules


@dataclass(frozen=True)
class Group:
    """Satu dimensi = FIS Mamdani dari dua kriteria."""

    name: str
    default_weight: float
    a: Criterion
    b: Criterion

    @property
    def criteria(self) -> tuple[Criterion, Criterion]:
        return (self.a, self.b)

    @property
    def rules(self) -> list[tuple[str, str, str]]:
        return _gen_rules(self.a, self.b)


# --------------------------------------------------------------------------- #
# Konfigurasi kriteria & kelompok
# --------------------------------------------------------------------------- #
CRITS = {
    "depression": Criterion("depression", "Tingkat Depresi", "benefit", 0, 27,
                            "Skor depresi (PHQ-style). Makin tinggi makin prioritas."),
    "anxiety_level": Criterion("anxiety_level", "Tingkat Kecemasan", "benefit", 0, 21,
                               "Skor kecemasan (GAD-style)."),
    "study_load": Criterion("study_load", "Tekanan Akademik", "benefit", 0, 5,
                            "Beban / tekanan studi yang dirasakan."),
    "future_career_concerns": Criterion("future_career_concerns", "Kekhawatiran Karier",
                                         "benefit", 0, 5, "Kekhawatiran masa depan karier."),
    "sleep_quality": Criterion("sleep_quality", "Kualitas Tidur", "cost", 0, 5,
                               "Kualitas tidur. Makin BURUK (rendah) makin prioritas."),
    "social_support": Criterion("social_support", "Dukungan Sosial", "cost", 0, 3,
                                "Dukungan sosial. Makin RENDAH makin prioritas."),
}

GROUPS = [
    Group("Psikologis", 0.45, CRITS["depression"], CRITS["anxiety_level"]),
    Group("Akademik", 0.30, CRITS["study_load"], CRITS["future_career_concerns"]),
    Group("Pendukung", 0.25, CRITS["sleep_quality"], CRITS["social_support"]),
]

ALL_CRITS = [c for g in GROUPS for c in g.criteria]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    file: str
    id_label: str
    label_col: str | None = None


DATASET = DatasetConfig("StressLevelDataset", "StressLevelDataset.csv", "MHS", "stress_level")


# --------------------------------------------------------------------------- #
# Pemuatan data
# --------------------------------------------------------------------------- #
def load_dataset(cfg: DatasetConfig, base_dir) -> pd.DataFrame:
    return pd.read_csv(f"{base_dir}/{cfg.file}")


# --------------------------------------------------------------------------- #
# Kurva keanggotaan (untuk plot)
# --------------------------------------------------------------------------- #
def membership_curves(crit: Criterion, n: int = 200):
    x = np.linspace(crit.vmin, crit.vmax, n)
    return x, {t: fuzz.trapmf(x, p) for t, p in crit.mf_params.items()}


def output_curves():
    """Semesta + fungsi keanggotaan keluaran prioritas (0-100)."""
    return Z, OUT_MF


def membership_degree(crit: Criterion, value: float) -> dict[str, float]:
    return {t: float(fuzz.trapmf(np.array([value]), p)[0]) for t, p in crit.mf_params.items()}


# --------------------------------------------------------------------------- #
# Inferensi Mamdani satu dimensi (vektorisasi atas seluruh baris)
# --------------------------------------------------------------------------- #
def _group_inference(group: Group, df: pd.DataFrame):
    """Return (skor_crisp[n], mu_a, mu_b) — derajat keanggotaan tiap term per input."""
    ca, cb = group.a, group.b
    va = np.clip(df[ca.key].to_numpy(float), ca.vmin, ca.vmax)
    vb = np.clip(df[cb.key].to_numpy(float), cb.vmin, cb.vmax)
    mu_a = {t: fuzz.trapmf(va, p) for t, p in ca.mf_params.items()}
    mu_b = {t: fuzz.trapmf(vb, p) for t, p in cb.mf_params.items()}

    n = len(df)
    agg = np.zeros((n, Z.size))
    for ta, tb, tout in group.rules:
        alpha = np.minimum(mu_a[ta], mu_b[tb])                 # (n,) firing strength
        clipped = np.minimum(alpha[:, None], OUT_MF[tout][None, :])  # (n, Z)
        agg = np.maximum(agg, clipped)                          # komposisi MAX

    num = (agg * Z[None, :]).sum(axis=1)
    den = agg.sum(axis=1) + 1e-9
    skor = num / den                                            # centroid
    return skor, mu_a, mu_b


def rule_firing(group: Group, values: dict[str, float]) -> pd.DataFrame:
    """Untuk SATU mahasiswa: tabel alpha tiap aturan (untuk inspeksi transparansi)."""
    ca, cb = group.a, group.b
    ma = membership_degree(ca, values[ca.key])
    mb = membership_degree(cb, values[cb.key])
    rows = []
    for ta, tb, tout in group.rules:
        rows.append({
            f"{ca.label}": ta, f"{cb.label}": tb, "→ Prioritas": tout,
            "α (firing)": round(min(ma[ta], mb[tb]), 3),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Perhitungan penuh
# --------------------------------------------------------------------------- #
def compute(df: pd.DataFrame, weights: dict[str, float]) -> dict[str, pd.DataFrame]:
    n = len(df)
    ids = np.array([f"{DATASET.id_label}-{i + 1:04d}" for i in range(n)])

    raw_w = np.array([weights[g.name] for g in GROUPS], dtype=float)
    if raw_w.sum() == 0:
        raw_w = np.array([g.default_weight for g in GROUPS])
    norm_w = raw_w / raw_w.sum()

    group_scores: dict[str, np.ndarray] = {}
    fuzz_cols: dict[str, np.ndarray] = {}
    raw_cols: dict[str, np.ndarray] = {}
    final = np.zeros(n)

    for g, w in zip(GROUPS, norm_w):
        skor, mu_a, mu_b = _group_inference(g, df)
        group_scores[g.name] = np.round(skor, 2)
        final += w * skor
        for crit, mu in [(g.a, mu_a), (g.b, mu_b)]:
            raw_cols[crit.label] = df[crit.key].to_numpy()
            for t in TERMS:
                fuzz_cols[f"{crit.label} · {t}"] = np.round(mu[t], 3)

    skor_100 = np.round(final, 2)
    kategori = np.array([_kategori(s) for s in skor_100])

    ranking = pd.DataFrame({"ID Mahasiswa": ids, **raw_cols})
    ranking["Skor Prioritas"] = skor_100
    ranking["Kategori"] = kategori
    ranking = ranking.sort_values("Skor Prioritas", ascending=False).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, n + 1))
    order = ranking["ID Mahasiswa"].to_numpy()

    fuzzifikasi = _reorder(pd.DataFrame({"ID Mahasiswa": ids, **fuzz_cols}), order)

    defuzz = pd.DataFrame({"ID Mahasiswa": ids})
    for name, sc in group_scores.items():
        defuzz[f"Skor {name}"] = sc
    defuzz["Skor Prioritas"] = skor_100
    defuzz = _reorder(defuzz, order)

    bobot = pd.DataFrame({
        "Dimensi": [g.name for g in GROUPS],
        "Kriteria": [f"{g.a.label} + {g.b.label}" for g in GROUPS],
        "Bobot Input": np.round(raw_w, 3),
        "Bobot Ternormalisasi": np.round(norm_w, 4),
    })

    return {
        "ranking": ranking,
        "fuzzifikasi": fuzzifikasi,
        "defuzzifikasi": defuzz,
        "bobot": bobot,
    }


def rule_base_table() -> pd.DataFrame:
    """Seluruh basis aturan (27) untuk ditampilkan di UI."""
    rows = []
    for g in GROUPS:
        for ta, tb, tout in g.rules:
            rows.append({
                "Dimensi": g.name,
                f"IF antiseden-1": f"{g.a.label} = {ta}",
                f"AND antiseden-2": f"{g.b.label} = {tb}",
                "THEN Prioritas": tout,
            })
    return pd.DataFrame(rows)


def _reorder(df: pd.DataFrame, order: np.ndarray) -> pd.DataFrame:
    return df.set_index("ID Mahasiswa").loc[order].reset_index()


def _kategori(skor: float) -> str:
    for ambang, label in KATEGORI_AMBANG:
        if skor >= ambang:
            return label
    return "Rendah"
