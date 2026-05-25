"""
SPK Pemilihan Prioritas Penanganan Kesehatan Mental Mahasiswa
Metode: Fuzzy MADM (Fuzzy-SAW)  |  Antarmuka: Streamlit

Jalankan:  uv run streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fuzzy_engine import (
    DATASET,
    DATASETS,
    compute,
    load_dataset,
    membership_curves,
    membership_degree,
)

BASE_DIR = Path(__file__).parent

# Dataset tunggal aplikasi.
ds_name = DATASET.name
cfg = DATASET

# --------------------------------------------------------------------------- #
# Palet & konfigurasi halaman
# --------------------------------------------------------------------------- #
C = {
    "primary": "#2A9D8F",
    "primary_dark": "#1F7A6F",
    "ink": "#16242B",
    "muted": "#5C6B73",
    "line": "#E3E9EC",
    "tinggi": "#E76F51",
    "sedang": "#E9C46A",
    "rendah": "#8AB17D",
    "bg_soft": "#F1F5F7",
}
KAT_COLOR = {"Tinggi": C["tinggi"], "Sedang": C["sedang"], "Rendah": C["rendah"]}

st.set_page_config(
    page_title="SPK Kesehatan Mental Mahasiswa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1280px; }}
      #MainMenu, footer {{ visibility: hidden; }}

      .hero {{
        background: linear-gradient(120deg, {C['primary_dark']} 0%, {C['primary']} 100%);
        color: #fff; border-radius: 16px; padding: 26px 30px; margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(31,122,111,.18);
      }}
      .hero h1 {{ font-size: 1.7rem; margin: 0 0 6px 0; font-weight: 700; letter-spacing:-.3px; }}
      .hero p  {{ margin: 0; opacity: .92; font-size: .96rem; max-width: 880px; }}
      .hero .pill {{
        display:inline-block; background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
        padding: 3px 11px; border-radius: 999px; font-size: .72rem; margin-right:7px; margin-top:12px;
        letter-spacing:.3px;
      }}

      .card {{
        background:#fff; border:1px solid {C['line']}; border-radius:13px; padding:16px 18px;
        box-shadow: 0 1px 2px rgba(22,36,43,.04); height:100%;
      }}
      .card .lbl {{ color:{C['muted']}; font-size:.74rem; text-transform:uppercase; letter-spacing:.6px; font-weight:600; }}
      .card .val {{ color:{C['ink']}; font-size:1.62rem; font-weight:700; line-height:1.15; margin-top:4px; }}
      .card .sub {{ color:{C['muted']}; font-size:.78rem; margin-top:2px; }}
      .card .bar {{ height:4px; border-radius:4px; margin-top:11px; }}

      .sec {{ font-size:1.12rem; font-weight:700; color:{C['ink']}; margin: 6px 0 2px; }}
      .sec-sub {{ color:{C['muted']}; font-size:.85rem; margin-bottom:10px; }}

      .badge {{ padding:2px 10px; border-radius:999px; color:#fff; font-size:.74rem; font-weight:600; }}

      .member {{
        background:#fff; border:1px solid {C['line']}; border-left:5px solid {C['primary']};
        border-radius:12px; padding:16px 18px;
      }}
      .member .nm {{ font-weight:700; color:{C['ink']}; font-size:1.05rem; }}
      .member .nim {{ color:{C['muted']}; font-size:.86rem; font-family:monospace; margin-top:2px; }}
      .member .rl {{ color:{C['primary_dark']}; font-size:.8rem; margin-top:6px; font-weight:600; }}

      div[data-testid="stSidebarUserContent"] {{ padding-top: 1rem; }}
      .stRadio [role="radiogroup"] label {{ font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Util
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load(name: str) -> pd.DataFrame:
    return load_dataset(DATASETS[name], BASE_DIR.as_posix())


def hero(title: str, subtitle: str, pills: list[str]):
    pill_html = "".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{pill_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(col, label, value, sub="", color=None):
    color = color or C["primary"]
    col.markdown(
        f'<div class="card"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div>'
        f'<div class="bar" style="background:{color}"></div></div>',
        unsafe_allow_html=True,
    )


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)


def base_layout(fig, h=360, legend_top=True):
    fig.update_layout(
        height=h,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=C["ink"], size=12),
        legend=dict(orientation="h", y=1.12, x=0) if legend_top else {},
    )
    fig.update_xaxes(gridcolor=C["line"], zeroline=False)
    fig.update_yaxes(gridcolor=C["line"], zeroline=False)
    return fig


def membership_figure(crit, value=None, value_label=None):
    """Plot kurva keanggotaan trapesium satu kriteria (+ penanda nilai opsional)."""
    x, curves = membership_curves(crit)
    cmap = {"Rendah": C["rendah"], "Sedang": C["sedang"], "Tinggi": C["tinggi"]}
    fillmap = {"Rendah": "rgba(138,177,125,.12)", "Sedang": "rgba(233,196,106,.12)",
               "Tinggi": "rgba(231,111,81,.12)"}
    fig = go.Figure()
    for lbl, y in curves.items():
        fig.add_trace(go.Scatter(x=x, y=y, name=lbl, mode="lines",
                                 line=dict(color=cmap[lbl], width=2.6),
                                 fill="tozeroy", fillcolor=fillmap[lbl]))
    if value is not None:
        fig.add_vline(x=value, line_dash="dash", line_color=C["ink"],
                      annotation_text=(value_label or f"{value:g}"))
    fig.update_layout(yaxis_title="Derajat keanggotaan μ", xaxis_title=crit.label,
                      yaxis_range=[0, 1.06])
    return base_layout(fig, 380)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        f'<div style="font-weight:800;font-size:1.05rem;color:{C["primary_dark"]}">🧠 SPK Kesehatan Mental</div>'
        f'<div style="color:{C["muted"]};font-size:.78rem;margin-bottom:14px">Fuzzy MADM · Streamlit</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigasi", ["Halaman Data", "Hitung SPK", "Profil Kelompok"],
                    label_visibility="collapsed", key="nav")
    st.divider()
    st.markdown(f"**Dataset**  \n`{cfg.file}`")
    st.caption(f"{len(cfg.criteria)} kriteria SPK")
    st.divider()
    st.caption("Metode Fuzzy-SAW: fuzzifikasi trapesium → "
               "defuzzifikasi centroid → agregasi bobot → perangkingan.")

df = _load(ds_name)


# =========================================================================== #
# HALAMAN DATA
# =========================================================================== #
def page_data():
    hero(
        "Eksplorasi Data Kesehatan Mental Mahasiswa",
        "Dataset mentah, profil kriteria, dan analisis distribusi sebagai dasar "
        "penilaian prioritas penanganan.",
        [f"{len(df):,} baris".replace(",", "."), f"{len(cfg.criteria)} kriteria SPK", ds_name],
    )

    # --- Ringkasan ---
    crit_keys = [c.key for c in cfg.criteria]
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Jumlah Mahasiswa", f"{len(df):,}".replace(",", "."),
                "alternatif keputusan")
    metric_card(c2, "Kriteria SPK", f"{len(cfg.criteria)}", "min. syarat: 5", C["primary_dark"])

    if cfg.label_col and cfg.label_col in df.columns:
        lbl = pd.to_numeric(df[cfg.label_col], errors="coerce")
        hi_share = (lbl >= lbl.max()).mean() * 100
        metric_card(c3, f"Label '{cfg.label_col}' Maks",
                    f"{hi_share:.0f}%", "porsi kategori tertinggi", C["tinggi"])
    else:
        metric_card(c3, "Label Target", "—", "tidak tersedia", C["muted"])

    first = cfg.criteria[0]
    metric_card(c4, f"Rata-rata {first.label}",
                f"{df[first.key].mean():.1f}", f"skala {first.vmin:.0f}–{first.vmax:.0f}",
                C["sedang"])

    st.write("")
    section("Dataset Mentah", "Tabel interaktif — klik header untuk mengurutkan.")
    st.dataframe(df, width="stretch", height=300)

    # --- Tabel kriteria ---
    section("Definisi Kriteria")
    crit_df = pd.DataFrame({
        "Kriteria": [c.label for c in cfg.criteria],
        "Kolom": [c.key for c in cfg.criteria],
        "Arah": [("Benefit ↑ prioritas" if c.direction == "benefit"
                  else "Cost ↓ prioritas") for c in cfg.criteria],
        "Rentang": [f"{c.vmin:.0f} – {c.vmax:.0f}" for c in cfg.criteria],
        "Bobot Default": [c.default_weight for c in cfg.criteria],
        "Keterangan": [c.keterangan for c in cfg.criteria],
    })
    st.dataframe(crit_df, width="stretch", hide_index=True)

    # --- Visualisasi ---
    st.write("")
    section("Analisis Visual")
    t1, t2, t3, t4, t5 = st.tabs(["Distribusi Kriteria", "Korelasi", "Sebaran",
                                  "Komposisi Target", "Fungsi Keanggotaan"])

    with t1:
        a, b = st.columns([1, 3])
        pick = a.selectbox("Kriteria", crit_keys,
                           format_func=lambda k: next((c.label for c in cfg.criteria if c.key == k), k))
        a.caption("Histogram nilai kriteria pada seluruh mahasiswa.")
        fig = px.histogram(df, x=pick, nbins=24, color_discrete_sequence=[C["primary"]])
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        mean_v = df[pick].mean()
        fig.add_vline(x=mean_v, line_dash="dash", line_color=C["tinggi"],
                      annotation_text=f"rata-rata {mean_v:.1f}", annotation_position="top")
        b.plotly_chart(base_layout(fig, 340, legend_top=False), width="stretch")

    with t2:
        st.caption("Korelasi Pearson antar kriteria"
                   + (" dan label target." if cfg.label_col in df.columns else "."))
        cols = crit_keys + ([cfg.label_col] if cfg.label_col in df.columns else [])
        corr = df[cols].corr().round(2)
        labels = {c.key: c.label for c in cfg.criteria}
        names = [labels.get(c, c) for c in cols]
        fig = px.imshow(corr, x=names, y=names, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(base_layout(fig, 460, legend_top=False), width="stretch")

    with t3:
        a, b, c = st.columns(3)
        xk = a.selectbox("Sumbu X", crit_keys, index=0,
                         format_func=lambda k: next((c.label for c in cfg.criteria if c.key == k), k))
        yk = b.selectbox("Sumbu Y", crit_keys, index=min(1, len(crit_keys) - 1),
                         format_func=lambda k: next((c.label for c in cfg.criteria if c.key == k), k))
        color_col = cfg.label_col if cfg.label_col in df.columns else None
        sample = df.sample(min(2500, len(df)), random_state=7)
        fig = px.scatter(sample, x=xk, y=yk,
                         color=color_col, color_continuous_scale="Teal", opacity=0.6)
        c.caption("Maks. 2.500 titik (sampel) agar grafik tetap responsif.")
        st.plotly_chart(base_layout(fig, 420, legend_top=False), width="stretch")

    with t4:
        if cfg.label_col and cfg.label_col in df.columns:
            vc = df[cfg.label_col].value_counts().sort_index()
            fig = px.bar(x=vc.index.astype(str), y=vc.values,
                         color=vc.values, color_continuous_scale="Teal",
                         labels={"x": cfg.label_col, "y": "Jumlah"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(base_layout(fig, 360, legend_top=False), width="stretch")
            st.caption("Distribusi label target asli dataset (bukan hasil SPK).")
        else:
            st.info("Dataset ini tidak memiliki kolom label target.")

    with t5:
        a, b = st.columns([1, 3])
        mk = a.selectbox("Kriteria", crit_keys, key="mf_data",
                         format_func=lambda k: next((c.label for c in cfg.criteria if c.key == k), k))
        crit = next(c for c in cfg.criteria if c.key == mk)
        a.caption(f"Himpunan fuzzy **trapesium**: Rendah · Sedang · Tinggi. "
                  f"Arah kriteria: **{crit.direction}** "
                  f"({'tinggi → prioritas' if crit.direction == 'benefit' else 'rendah → prioritas'}).")
        a.caption("Inilah dasar fuzzifikasi: tiap nilai dipetakan ke derajat "
                  "keanggotaan tiap himpunan.")
        b.plotly_chart(membership_figure(crit), width="stretch")


# =========================================================================== #
# HITUNG SPK
# =========================================================================== #
def page_spk():
    hero(
        "Perhitungan Prioritas — Fuzzy MADM (SAW)",
        "Atur bobot kriteria, lalu jalankan perhitungan untuk menghasilkan "
        "peringkat mahasiswa berdasarkan urgensi penanganan kesehatan mental.",
        ["Bobot dinamis", "Kurva keanggotaan", "Defuzzifikasi centroid"],
    )

    with st.expander("Bagaimana metode ini bekerja?"):
        st.markdown(
            "1. **Fuzzifikasi** — tiap nilai kriteria diubah jadi derajat keanggotaan "
            "*Rendah / Sedang / Tinggi* memakai fungsi segitiga (`skfuzzy.trimf`).\n"
            "2. **Defuzzifikasi** — derajat keanggotaan diringkas jadi skor *severity* 0–1 "
            "via centroid berbobot. Kriteria **cost** (mis. kualitas tidur, dukungan sosial) "
            "dibalik: nilai rendah → severity tinggi.\n"
            "3. **Agregasi SAW** — `skor = Σ(bobot × severity)`, bobot dinormalisasi otomatis.\n"
            "4. **Perangkingan** — skor diurut menurun; Peringkat 1 = prioritas tertinggi."
        )

    # --- Panel bobot ---
    section("1 · Atur Bobot Kriteria")
    left, right = st.columns([3, 2])
    weights = {}
    with left:
        for c in cfg.criteria:
            arah = "↑ benefit" if c.direction == "benefit" else "↓ cost"
            weights[c.key] = st.slider(
                f"{c.label}  ·  {arah}", 0.0, 1.0, float(c.default_weight), 0.01,
                help=c.keterangan,
            )
    with right:
        raw = np.array(list(weights.values()))
        norm = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw) / len(raw)
        donut = go.Figure(go.Pie(
            labels=[c.label for c in cfg.criteria], values=norm, hole=.58,
            marker=dict(colors=px.colors.sequential.Teal[-len(cfg.criteria):]),
            textinfo="percent", sort=False,
        ))
        donut.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                            title="Bobot Ternormalisasi", showlegend=True,
                            legend=dict(orientation="v", x=1, y=.5, font=dict(size=10)))
        st.plotly_chart(donut, width="stretch")

    # --- Kontrol eksekusi ---
    section("2 · Jalankan Perhitungan")
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    top_n = cc1.number_input("Tampilkan Top-N", 5, 100, 15, 5)
    kat_filter = cc2.selectbox("Filter Kategori", ["Semua", "Tinggi", "Sedang", "Rendah"])
    run = cc3.button("🔎  Hitung Prioritas", type="primary", width="stretch", key="run_btn")

    if run:
        st.session_state["hasil"] = compute(df, cfg, weights)
        st.session_state["hasil_ds"] = ds_name

    if "hasil" not in st.session_state or st.session_state.get("hasil_ds") != ds_name:
        st.info("Atur bobot lalu tekan **Hitung Prioritas** untuk melihat hasil perangkingan.")
        return

    res = st.session_state["hasil"]
    ranking, fz, dfz, bobot = res["ranking"], res["fuzzifikasi"], res["defuzzifikasi"], res["bobot"]

    view = ranking if kat_filter == "Semua" else ranking[ranking["Kategori"] == kat_filter]
    view_top = view.head(top_n)

    # --- Ringkasan hasil ---
    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "Dievaluasi", f"{len(ranking):,}".replace(",", "."), "total mahasiswa")
    metric_card(m2, "Skor Tertinggi", f"{ranking['Skor Prioritas'].iloc[0]:.1f}",
                f"ID {ranking['ID Mahasiswa'].iloc[0]}", C["tinggi"])
    metric_card(m3, "Rata-rata Skor", f"{ranking['Skor Prioritas'].mean():.1f}",
                "skala 0–100", C["primary_dark"])
    metric_card(m4, "Prioritas Tinggi",
                f"{(ranking['Kategori'] == 'Tinggi').sum():,}".replace(",", "."),
                "skor ≥ 66", C["tinggi"])

    # --- Tabel ranking ---
    st.write("")
    section("3 · Tabel Hasil Perangkingan",
            f"Top-{top_n} dari {len(view):,} mahasiswa".replace(",", "."))

    show_cols = ["Peringkat", "ID Mahasiswa"] + [c.label for c in cfg.criteria] \
                + ["Skor Prioritas", "Kategori"]
    styler = (
        view_top[show_cols].style
        .background_gradient(subset=["Skor Prioritas"], cmap="YlOrRd")
        .apply(lambda s: [f"color:{KAT_COLOR.get(v,'#000')};font-weight:700" for v in s],
               subset=["Kategori"])
        .format({"Skor Prioritas": "{:.2f}"})
    )
    st.dataframe(styler, width="stretch", hide_index=True,
                 height=min(60 + 35 * len(view_top), 560))
    st.download_button("⬇️ Unduh ranking lengkap (CSV)",
                       ranking.to_csv(index=False).encode("utf-8"),
                       file_name=f"ranking_{ds_name}.csv", mime="text/csv")

    # --- Bar chart top-N ---
    st.write("")
    bar = px.bar(view_top.sort_values("Skor Prioritas"),
                 x="Skor Prioritas", y="ID Mahasiswa", orientation="h",
                 color="Kategori", color_discrete_map=KAT_COLOR, text="Skor Prioritas")
    bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    bar.update_layout(yaxis_title="", xaxis_range=[0, 100])
    st.plotly_chart(base_layout(bar, max(340, 26 * len(view_top))), width="stretch")

    # --- Detail fuzzy ---
    st.write("")
    section("4 · Proses Fuzzy (Transparansi Perhitungan)")
    tk, tf, td, tr, ts = st.tabs(
        ["Kurva Keanggotaan", "Fuzzifikasi", "Defuzzifikasi", "Radar Profil", "Sebaran Skor"]
    )

    ids_top = view_top["ID Mahasiswa"].tolist()

    with tk:
        a, b = st.columns([1, 1])
        ck = a.selectbox("Kriteria", [c.key for c in cfg.criteria],
                         format_func=lambda k: next((c.label for c in cfg.criteria if c.key == k), k))
        sid = b.selectbox("Soroti mahasiswa", ids_top)
        crit = next(c for c in cfg.criteria if c.key == ck)
        val = float(ranking.loc[ranking["ID Mahasiswa"] == sid, crit.label].iloc[0])
        deg = membership_degree(crit, val)
        st.plotly_chart(membership_figure(crit, val, f"{sid} = {val:g}"), width="stretch")
        deg_txt = " · ".join(f"{k}: **{v:.2f}**" for k, v in deg.items())
        st.caption(f"Derajat keanggotaan {sid} pada {crit.label} → {deg_txt}  "
                   f"(arah **{crit.direction}**)")

    with tf:
        st.caption("Derajat keanggotaan tiap kriteria × {Rendah, Sedang, Tinggi} "
                   "untuk Top-N mahasiswa.")
        st.dataframe(fz.head(top_n), width="stretch", hide_index=True)

    with td:
        st.caption("Skor severity 0–1 hasil defuzzifikasi (centroid) per kriteria, "
                   "lalu skor prioritas akhir.")
        dview = dfz.head(top_n)
        sev_cols = [c.label for c in cfg.criteria]
        styler2 = (dview.style
                   .background_gradient(subset=sev_cols, cmap="YlOrRd", vmin=0, vmax=1)
                   .background_gradient(subset=["Skor Prioritas"], cmap="YlOrRd")
                   .format({**{c: "{:.3f}" for c in sev_cols}, "Skor Prioritas": "{:.2f}"}))
        st.dataframe(styler2, width="stretch", hide_index=True)

    with tr:
        sid2 = st.selectbox("Pilih mahasiswa", ids_top, key="radar_id")
        row = dfz[dfz["ID Mahasiswa"] == sid2].iloc[0]
        cats = [c.label for c in cfg.criteria]
        vals = [row[c] for c in cats]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
            line=dict(color=C["primary"]), fillcolor="rgba(42,157,143,.25)", name=sid2))
        radar.update_layout(height=420, polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
                            margin=dict(l=40, r=40, t=40, b=40),
                            title=f"Profil severity · {sid2}")
        st.plotly_chart(radar, width="stretch")
        st.caption("Nilai mendekati 1 = kriteria tersebut menekan kuat ke arah prioritas tinggi.")

    with ts:
        a, b = st.columns([3, 2])
        h = px.histogram(ranking, x="Skor Prioritas", nbins=30,
                         color="Kategori", color_discrete_map=KAT_COLOR)
        h.update_layout(xaxis_range=[0, 100], barmode="stack")
        a.plotly_chart(base_layout(h, 360), width="stretch")
        kv = ranking["Kategori"].value_counts().reindex(["Tinggi", "Sedang", "Rendah"]).fillna(0)
        pie = go.Figure(go.Pie(labels=kv.index, values=kv.values, hole=.5,
                               marker=dict(colors=[KAT_COLOR[k] for k in kv.index]),
                               textinfo="label+percent", sort=False))
        pie.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0),
                         title="Komposisi Kategori", showlegend=False)
        b.plotly_chart(pie, width="stretch")


# =========================================================================== #
# PROFIL KELOMPOK
# =========================================================================== #
def page_profil():
    hero(
        "Profil Kelompok",
        "Proyek Akhir Praktikum Sistem Pendukung Pengambilan Keputusan (SCPK) 2025/2026.",
        ["Metode Fuzzy", "Streamlit", "SCPK 2025/2026"],
    )

    section("Anggota Kelompok")
    members = [
        ("Bintang Ramadhan", "123240073", "Engine Fuzzy & SPK"),
        ("Arsyadi Indra Hasan P.", "123240062", "Antarmuka & Analisis"),
    ]
    cols = st.columns(len(members))
    for col, (nm, nim, rl) in zip(cols, members):
        col.markdown(
            f'<div class="member"><div class="nm">{nm}</div>'
            f'<div class="nim">{nim}</div><div class="rl">{rl}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    section("Informasi Proyek")
    info = pd.DataFrame({
        "Item": ["Judul", "Metode SPK", "Dataset", "Jumlah Kriteria",
                 "Mata Kuliah", "Antarmuka"],
        "Keterangan": [
            "Pemilihan Prioritas Penanganan Kesehatan Mental Mahasiswa",
            "Fuzzy MADM (Fuzzy-SAW): fuzzifikasi segitiga + defuzzifikasi centroid",
            f"{cfg.name} ({cfg.file})",
            f"{len(cfg.criteria)} kriteria (memenuhi syarat ≥ 5)",
            "Praktikum SCPK 2025/2026",
            "Streamlit (navigasi sidebar, bobot dinamis, tombol eksekusi)",
        ],
    })
    st.dataframe(info, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
if page == "Halaman Data":
    page_data()
elif page == "Hitung SPK":
    page_spk()
else:
    page_profil()
