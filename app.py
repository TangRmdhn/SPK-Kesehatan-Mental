"""
SPK Pemilihan Prioritas Penanganan Kesehatan Mental Mahasiswa
Metode: Fuzzy Inference System (Mamdani) hierarkis + agregasi bobot dimensi
Antarmuka: Streamlit  |  Jalankan: uv run streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fuzzy_engine import (
    ALL_CRITS,
    DATASET,
    GROUPS,
    compute,
    load_dataset,
    membership_curves,
    membership_degree,
    output_curves,
    rule_base_table,
    rule_firing,
)

BASE_DIR = Path(__file__).parent
ds_name = DATASET.name
cfg = DATASET

# --------------------------------------------------------------------------- #
# Palet & konfigurasi
# --------------------------------------------------------------------------- #
C = {
    "primary": "#2A9D8F", "primary_dark": "#1F7A6F", "ink": "#16242B",
    "muted": "#5C6B73", "line": "#E3E9EC",
    "tinggi": "#E76F51", "sedang": "#E9C46A", "rendah": "#8AB17D", "bg_soft": "#F1F5F7",
}
KAT_COLOR = {"Tinggi": C["tinggi"], "Sedang": C["sedang"], "Rendah": C["rendah"]}
TERM_COLOR = {"Rendah": C["rendah"], "Sedang": C["sedang"], "Tinggi": C["tinggi"]}
TERM_FILL = {"Rendah": "rgba(138,177,125,.12)", "Sedang": "rgba(233,196,106,.12)",
             "Tinggi": "rgba(231,111,81,.12)"}

st.set_page_config(page_title="SPK Kesehatan Mental Mahasiswa",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1280px; }}
      #MainMenu, footer {{ visibility: hidden; }}
      .hero {{ background: linear-gradient(120deg, {C['primary_dark']} 0%, {C['primary']} 100%);
        color:#fff; border-radius:16px; padding:26px 30px; margin-bottom:22px;
        box-shadow:0 8px 24px rgba(31,122,111,.18); }}
      .hero h1 {{ font-size:1.7rem; margin:0 0 6px 0; font-weight:700; letter-spacing:-.3px; }}
      .hero p {{ margin:0; opacity:.92; font-size:.96rem; max-width:880px; }}
      .hero .pill {{ display:inline-block; background:rgba(255,255,255,.16);
        border:1px solid rgba(255,255,255,.28); padding:3px 11px; border-radius:999px;
        font-size:.72rem; margin-right:7px; margin-top:12px; letter-spacing:.3px; }}
      .card {{ background:#fff; border:1px solid {C['line']}; border-radius:13px;
        padding:16px 18px; box-shadow:0 1px 2px rgba(22,36,43,.04); height:100%; }}
      .card .lbl {{ color:{C['muted']}; font-size:.74rem; text-transform:uppercase;
        letter-spacing:.6px; font-weight:600; }}
      .card .val {{ color:{C['ink']}; font-size:1.62rem; font-weight:700; line-height:1.15; margin-top:4px; }}
      .card .sub {{ color:{C['muted']}; font-size:.78rem; margin-top:2px; }}
      .card .bar {{ height:4px; border-radius:4px; margin-top:11px; }}
      .sec {{ font-size:1.12rem; font-weight:700; color:{C['ink']}; margin:6px 0 2px; }}
      .sec-sub {{ color:{C['muted']}; font-size:.85rem; margin-bottom:10px; }}
      .member {{ background:#fff; border:1px solid {C['line']}; border-left:5px solid {C['primary']};
        border-radius:12px; padding:16px 18px; }}
      .member .nm {{ font-weight:700; color:{C['ink']}; font-size:1.05rem; }}
      .member .nim {{ color:{C['muted']}; font-size:.86rem; font-family:monospace; margin-top:2px; }}
      .member .rl {{ color:{C['primary_dark']}; font-size:.8rem; margin-top:6px; font-weight:600; }}
      .stRadio [role="radiogroup"] label {{ font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Util
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    return load_dataset(DATASET, BASE_DIR.as_posix())


def hero(title, subtitle, pills):
    pill_html = "".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{pill_html}</div>',
                unsafe_allow_html=True)


def metric_card(col, label, value, sub="", color=None):
    color = color or C["primary"]
    col.markdown(f'<div class="card"><div class="lbl">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div>'
                 f'<div class="bar" style="background:{color}"></div></div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)


def base_layout(fig, h=360, legend_top=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color=C["ink"], size=12),
                      legend=dict(orientation="h", y=1.12, x=0) if legend_top else {})
    fig.update_xaxes(gridcolor=C["line"], zeroline=False)
    fig.update_yaxes(gridcolor=C["line"], zeroline=False)
    return fig


def curve_figure(x, curves, xtitle, value=None, value_label=None):
    """Plot kurva keanggotaan (trapesium) + penanda nilai opsional."""
    fig = go.Figure()
    for lbl, y in curves.items():
        fig.add_trace(go.Scatter(x=x, y=y, name=lbl, mode="lines",
                                 line=dict(color=TERM_COLOR[lbl], width=2.6),
                                 fill="tozeroy", fillcolor=TERM_FILL[lbl]))
    if value is not None:
        fig.add_vline(x=value, line_dash="dash", line_color=C["ink"],
                      annotation_text=(value_label or f"{value:g}"))
    fig.update_layout(yaxis_title="Derajat keanggotaan μ", xaxis_title=xtitle,
                      yaxis_range=[0, 1.06])
    return base_layout(fig, 360)


CRIT_BY_KEY = {c.key: c for c in ALL_CRITS}


def clabel(k):
    return CRIT_BY_KEY[k].label if k in CRIT_BY_KEY else k


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        f'<div style="font-weight:800;font-size:1.05rem;color:{C["primary_dark"]}">SPK Kesehatan Mental</div>'
        f'<div style="color:{C["muted"]};font-size:.78rem;margin-bottom:14px">Fuzzy Mamdani · Streamlit</div>',
        unsafe_allow_html=True)
    page = st.radio("Navigasi", ["Halaman Data", "Hitung SPK", "Profil Kelompok"],
                    label_visibility="collapsed", key="nav")
    st.divider()
    st.markdown(f"**Dataset**  \n`{cfg.file}`")
    st.caption(f"{len(ALL_CRITS)} kriteria · {len(GROUPS)} dimensi FIS")
    st.divider()
    st.caption("FIS Mamdani hierarkis: 3 dimensi (9 aturan/dimensi) → "
               "defuzzifikasi centroid → agregasi bobot → perangkingan.")

df = _load()


# =========================================================================== #
# HALAMAN DATA
# =========================================================================== #
def page_data():
    hero("Eksplorasi Data Kesehatan Mental Mahasiswa",
         "Dataset mentah, profil kriteria, dan analisis distribusi sebagai dasar "
         "penilaian prioritas penanganan.",
         [f"{len(df):,} baris".replace(",", "."), f"{len(ALL_CRITS)} kriteria",
          f"{len(GROUPS)} dimensi FIS"])

    keys = [c.key for c in ALL_CRITS]
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Jumlah Mahasiswa", f"{len(df):,}".replace(",", "."), "alternatif keputusan")
    metric_card(c2, "Kriteria / Dimensi", f"{len(ALL_CRITS)} / {len(GROUPS)}",
                "6 kriteria, 3 FIS", C["primary_dark"])
    if cfg.label_col in df.columns:
        lbl = pd.to_numeric(df[cfg.label_col], errors="coerce")
        metric_card(c3, f"Label '{cfg.label_col}' Maks",
                    f"{(lbl >= lbl.max()).mean()*100:.0f}%", "porsi kategori tertinggi", C["tinggi"])
    first = ALL_CRITS[0]
    metric_card(c4, f"Rata-rata {first.label}", f"{df[first.key].mean():.1f}",
                f"skala {first.vmin:.0f}–{first.vmax:.0f}", C["sedang"])

    st.write("")
    section("Dataset Mentah", "Tabel interaktif — klik header untuk mengurutkan.")
    st.dataframe(df, width="stretch", height=300)

    section("Definisi Kriteria per Dimensi FIS")
    rows = []
    for g in GROUPS:
        for c in g.criteria:
            rows.append({"Dimensi": g.name, "Kriteria": c.label, "Kolom": c.key,
                         "Arah": "Benefit ↑" if c.direction == "benefit" else "Cost ↓",
                         "Rentang": f"{c.vmin:.0f} – {c.vmax:.0f}", "Keterangan": c.keterangan})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("")
    section("Analisis Visual")
    t1, t2, t3, t4, t5 = st.tabs(["Distribusi", "Korelasi", "Sebaran",
                                  "Komposisi Target", "Fungsi Keanggotaan"])
    with t1:
        a, b = st.columns([1, 3])
        pick = a.selectbox("Kriteria", keys, format_func=clabel)
        a.caption("Histogram nilai kriteria seluruh mahasiswa.")
        fig = px.histogram(df, x=pick, nbins=24, color_discrete_sequence=[C["primary"]])
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.add_vline(x=df[pick].mean(), line_dash="dash", line_color=C["tinggi"],
                      annotation_text=f"rata-rata {df[pick].mean():.1f}", annotation_position="top")
        b.plotly_chart(base_layout(fig, 340, legend_top=False), width="stretch")
    with t2:
        st.caption("Korelasi Pearson antar kriteria + label target.")
        cols = keys + ([cfg.label_col] if cfg.label_col in df.columns else [])
        corr = df[cols].corr().round(2)
        names = [clabel(c) for c in cols]
        fig = px.imshow(corr, x=names, y=names, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(base_layout(fig, 460, legend_top=False), width="stretch")
    with t3:
        a, b, c = st.columns(3)
        xk = a.selectbox("Sumbu X", keys, index=0, format_func=clabel)
        yk = b.selectbox("Sumbu Y", keys, index=1, format_func=clabel)
        color_col = cfg.label_col if cfg.label_col in df.columns else None
        sample = df.sample(min(2500, len(df)), random_state=7)
        fig = px.scatter(sample, x=xk, y=yk, color=color_col,
                         color_continuous_scale="Teal", opacity=0.6)
        c.caption("Maks. 2.500 titik (sampel).")
        st.plotly_chart(base_layout(fig, 420, legend_top=False), width="stretch")
    with t4:
        if cfg.label_col in df.columns:
            vc = df[cfg.label_col].value_counts().sort_index()
            fig = px.bar(x=vc.index.astype(str), y=vc.values, color=vc.values,
                         color_continuous_scale="Teal", labels={"x": cfg.label_col, "y": "Jumlah"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(base_layout(fig, 360, legend_top=False), width="stretch")
            st.caption("Distribusi label target asli (bukan hasil SPK).")
    with t5:
        a, b = st.columns([1, 3])
        opt = keys + ["__out__"]
        mk = a.selectbox("Variabel", opt, format_func=lambda k: "Keluaran: Prioritas"
                         if k == "__out__" else clabel(k), key="mf_data")
        if mk == "__out__":
            a.caption("Fungsi keanggotaan **keluaran** (prioritas 0–100) — dipakai saat defuzzifikasi.")
            x, curves = output_curves()
            b.plotly_chart(curve_figure(x, curves, "Skor Prioritas (0–100)"), width="stretch")
        else:
            crit = CRIT_BY_KEY[mk]
            a.caption(f"Fungsi keanggotaan **masukan**. Arah: **{crit.direction}**.")
            x, curves = membership_curves(crit)
            b.plotly_chart(curve_figure(x, curves, crit.label), width="stretch")


# =========================================================================== #
# HITUNG SPK
# =========================================================================== #
def page_spk():
    hero("Perhitungan Prioritas — Fuzzy Inference System (Mamdani)",
         "Atur bobot tiap dimensi, jalankan inferensi aturan, lalu lihat peringkat "
         "mahasiswa berdasarkan urgensi penanganan.",
         ["FIS hierarkis", "27 aturan IF-THEN", "Defuzzifikasi centroid", "Bobot dimensi"])

    with st.expander("Bagaimana metode FIS ini bekerja?"):
        st.markdown(
            "1. **Fuzzifikasi** — nilai tiap kriteria → derajat keanggotaan "
            "*Rendah/Sedang/Tinggi* (trapesium).\n"
            "2. **Inferensi** — 6 kriteria dibagi **3 dimensi** (2 kriteria/dimensi). "
            "Tiap dimensi = FIS Mamdani 9 aturan `IF a AND b THEN prioritas`. "
            "Kekuatan aturan `α = min(μ_a, μ_b)`.\n"
            "3. **Komposisi + Defuzzifikasi** — agregasi MAX konsekuen, lalu **centroid** "
            "→ skor crisp 0–100 tiap dimensi.\n"
            "4. **Agregasi bobot** — `skor akhir = Σ(bobot_dimensi × skor_dimensi)`.\n"
            "5. **Perangkingan** — urut menurun. Peringkat 1 = prioritas tertinggi.\n\n"
            "FIS hierarkis dipakai agar tak terjadi ledakan aturan (1 FIS 6 input = "
            "3⁶ = 729 aturan)."
        )

    section("1 · Bobot Antar-Dimensi")
    left, right = st.columns([3, 2])
    weights = {}
    with left:
        for g in GROUPS:
            weights[g.name] = st.slider(f"{g.name}  ·  {g.a.label} + {g.b.label}",
                                        0.0, 1.0, float(g.default_weight), 0.01, key=f"w_{g.name}")
    with right:
        raw = np.array(list(weights.values()))
        norm = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw) / len(raw)
        donut = go.Figure(go.Pie(labels=[g.name for g in GROUPS], values=norm, hole=.58,
                                 marker=dict(colors=[C["primary_dark"], C["primary"], C["sedang"]]),
                                 textinfo="percent", sort=False))
        donut.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                            title="Bobot Ternormalisasi", showlegend=True,
                            legend=dict(orientation="v", x=1, y=.5, font=dict(size=11)))
        st.plotly_chart(donut, width="stretch")

    section("2 · Basis Aturan (Rule Base)", "27 aturan IF-THEN — inti dari FIS.")
    rb = rule_base_table()
    rtabs = st.tabs([g.name for g in GROUPS])
    for tab, g in zip(rtabs, GROUPS):
        with tab:
            sub = rb[rb["Dimensi"] == g.name].drop(columns="Dimensi").reset_index(drop=True)
            st.dataframe(sub.style.apply(
                lambda s: [f"color:{KAT_COLOR.get(v,'#000')};font-weight:700" for v in s],
                subset=["THEN Prioritas"]), width="stretch", hide_index=True)

    section("3 · Jalankan Inferensi")
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    top_n = cc1.number_input("Tampilkan Top-N", 5, 100, 15, 5)
    kat_filter = cc2.selectbox("Filter Kategori", ["Semua", "Tinggi", "Sedang", "Rendah"])
    run = cc3.button("⚙️  Hitung Prioritas", type="primary", width="stretch", key="run_btn")
    if run:
        st.session_state["hasil"] = compute(df, weights)

    if "hasil" not in st.session_state:
        st.info("Atur bobot lalu tekan **Hitung Prioritas**.")
        return

    res = st.session_state["hasil"]
    ranking, fz, dfz = res["ranking"], res["fuzzifikasi"], res["defuzzifikasi"]
    view = ranking if kat_filter == "Semua" else ranking[ranking["Kategori"] == kat_filter]
    view_top = view.head(top_n)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "Dievaluasi", f"{len(ranking):,}".replace(",", "."), "total mahasiswa")
    metric_card(m2, "Skor Tertinggi", f"{ranking['Skor Prioritas'].iloc[0]:.1f}",
                f"ID {ranking['ID Mahasiswa'].iloc[0]}", C["tinggi"])
    metric_card(m3, "Rata-rata Skor", f"{ranking['Skor Prioritas'].mean():.1f}", "skala 0–100", C["primary_dark"])
    metric_card(m4, "Prioritas Tinggi", f"{(ranking['Kategori']=='Tinggi').sum():,}".replace(",", "."),
                "skor ≥ 66", C["tinggi"])

    st.write("")
    section("4 · Tabel Hasil Perangkingan", f"Top-{top_n} dari {len(view):,} mahasiswa".replace(",", "."))
    show = ["Peringkat", "ID Mahasiswa"] + [c.label for c in ALL_CRITS] + ["Skor Prioritas", "Kategori"]
    styler = (view_top[show].style
              .background_gradient(subset=["Skor Prioritas"], cmap="YlOrRd")
              .apply(lambda s: [f"color:{KAT_COLOR.get(v,'#000')};font-weight:700" for v in s],
                     subset=["Kategori"]).format({"Skor Prioritas": "{:.2f}"}))
    st.dataframe(styler, width="stretch", hide_index=True,
                 height=min(60 + 35 * len(view_top), 560))
    st.download_button("⬇️ Unduh ranking lengkap (CSV)",
                       ranking.to_csv(index=False).encode("utf-8"),
                       file_name="ranking_prioritas.csv", mime="text/csv")

    st.write("")
    bar = px.bar(view_top.sort_values("Skor Prioritas"), x="Skor Prioritas", y="ID Mahasiswa",
                 orientation="h", color="Kategori", color_discrete_map=KAT_COLOR, text="Skor Prioritas")
    bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    bar.update_layout(yaxis_title="", xaxis_range=[0, 100])
    st.plotly_chart(base_layout(bar, max(340, 26 * len(view_top))), width="stretch")

    st.write("")
    section("5 · Proses Fuzzy (Transparansi)")
    tk, tf, ti, td, ts = st.tabs(["Kurva Keanggotaan", "Fuzzifikasi", "Inferensi Aturan",
                                  "Defuzzifikasi", "Sebaran Skor"])
    ids_top = view_top["ID Mahasiswa"].tolist()

    with tk:
        a, b = st.columns([1, 1])
        ck = a.selectbox("Variabel", [c.key for c in ALL_CRITS] + ["__out__"],
                         format_func=lambda k: "Keluaran: Prioritas" if k == "__out__" else clabel(k))
        if ck == "__out__":
            x, curves = output_curves()
            b.plotly_chart(curve_figure(x, curves, "Skor Prioritas (0–100)"), width="stretch")
            st.caption("Fungsi keanggotaan keluaran prioritas (dipakai untuk defuzzifikasi centroid).")
        else:
            sid = a.selectbox("Soroti mahasiswa", ids_top)
            crit = CRIT_BY_KEY[ck]
            val = float(ranking.loc[ranking["ID Mahasiswa"] == sid, crit.label].iloc[0])
            deg = membership_degree(crit, val)
            x, curves = membership_curves(crit)
            b.plotly_chart(curve_figure(x, curves, crit.label, val, f"{sid} = {val:g}"), width="stretch")
            st.caption(f"Derajat keanggotaan {sid} pada {crit.label} → "
                       + " · ".join(f"{k}: **{v:.2f}**" for k, v in deg.items())
                       + f"  (arah **{crit.direction}**)")

    with tf:
        st.caption("Derajat keanggotaan tiap kriteria × {Rendah, Sedang, Tinggi} untuk Top-N.")
        st.dataframe(fz.head(top_n), width="stretch", hide_index=True)

    with ti:
        sid = st.selectbox("Pilih mahasiswa", ids_top, key="fire_id")
        idx = int(sid.split("-")[1]) - 1
        vals = {c.key: float(df.iloc[idx][c.key]) for c in ALL_CRITS}
        st.caption("Aturan yang **aktif** (α > 0) dan kekuatannya untuk mahasiswa ini, per dimensi. "
                   "Skor dimensi = centroid dari konsekuen aturan-aturan ini.")
        gtabs = st.tabs([f"{g.name} (skor {dfz.loc[dfz['ID Mahasiswa']==sid, f'Skor {g.name}'].iloc[0]:.1f})"
                         for g in GROUPS])
        for tab, g in zip(gtabs, GROUPS):
            with tab:
                rf = rule_firing(g, vals)
                active = rf[rf["α (firing)"] > 0].reset_index(drop=True)
                st.dataframe(active.style.apply(
                    lambda s: [f"color:{KAT_COLOR.get(v,'#000')};font-weight:700" for v in s],
                    subset=["→ Prioritas"]).background_gradient(subset=["α (firing)"], cmap="YlOrRd",
                    vmin=0, vmax=1), width="stretch", hide_index=True)

    with td:
        st.caption("Skor crisp tiap dimensi (hasil defuzzifikasi centroid) lalu skor akhir berbobot.")
        dview = dfz.head(top_n)
        sc_cols = [f"Skor {g.name}" for g in GROUPS]
        st.dataframe(dview.style
                     .background_gradient(subset=sc_cols + ["Skor Prioritas"], cmap="YlOrRd", vmin=0, vmax=100)
                     .format({c: "{:.2f}" for c in sc_cols + ["Skor Prioritas"]}),
                     width="stretch", hide_index=True)

    with ts:
        a, b = st.columns([3, 2])
        h = px.histogram(ranking, x="Skor Prioritas", nbins=30, color="Kategori",
                         color_discrete_map=KAT_COLOR)
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
    hero("Profil Kelompok",
         "Proyek Akhir Praktikum Sistem Pendukung Pengambilan Keputusan (SCPK) 2025/2026.",
         ["Metode Fuzzy Mamdani", "Streamlit", "SCPK 2025/2026"])

    section("Anggota Kelompok")
    members = [
        ("Bintang Ramadhan", "123240073", "Engine Fuzzy & SPK"),
        ("Arsyadi Indra Hasan P.", "123240062", "Antarmuka & Analisis"),
    ]
    cols = st.columns(len(members))
    for col, (nm, nim, rl) in zip(cols, members):
        col.markdown(f'<div class="member"><div class="nm">{nm}</div>'
                     f'<div class="nim">{nim}</div><div class="rl">{rl}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    section("Informasi Proyek")
    info = pd.DataFrame({
        "Item": ["Judul", "Metode SPK", "Dataset", "Kriteria / Dimensi",
                 "Jumlah Aturan", "Mata Kuliah", "Antarmuka"],
        "Keterangan": [
            "Pemilihan Prioritas Penanganan Kesehatan Mental Mahasiswa",
            "Fuzzy Inference System (Mamdani) hierarkis + agregasi bobot dimensi",
            f"{cfg.name} ({cfg.file})",
            f"{len(ALL_CRITS)} kriteria dalam {len(GROUPS)} dimensi FIS",
            f"{len(GROUPS) * 9} aturan IF-THEN (9 per dimensi)",
            "Praktikum SCPK 2025/2026",
            "Streamlit (navigasi sidebar, bobot dinamis, tombol eksekusi)",
        ],
    })
    st.dataframe(info, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
if page == "Halaman Data":
    page_data()
elif page == "Hitung SPK":
    page_spk()
else:
    page_profil()
