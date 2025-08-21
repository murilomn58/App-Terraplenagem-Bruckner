import streamlit as st
import pandas as pd
import numpy as np
import pulp
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from typing import List, Tuple

# ============================================================
# Utils
# ============================================================
def pos_to_estaca(x_m):
    """Converte metros em estacas (20 m = 1 estaca). Aceita escalar ou array-like."""
    vals = np.atleast_1d(x_m)
    out = []
    for v in vals:
        try:
            out.append(float(v) / 20.0)
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)

def infer_section_span_est(df_sorted: pd.DataFrame) -> float:
    """Estima o comprimento típico da seção (em estacas)."""
    if "pos_m" not in df_sorted or len(df_sorted) < 2:
        return 0.0
    step_m = float(pd.Series(df_sorted["pos_m"]).diff().median())
    if not np.isfinite(step_m) or step_m <= 0:
        return 0.0
    return round(step_m / 20.0, 1)

def _roman(n: int) -> str:
    if n <= 0:
        return str(n)
    table = [
        (1000,"M"), (900,"CM"), (500,"D"), (400,"CD"),
        (100,"C"), (90,"XC"), (50,"L"), (40,"XL"),
        (10,"X"), (9,"IX"), (5,"V"), (4,"IV"), (1,"I"),
    ]
    s = []
    for v, sym in table:
        while n >= v:
            s.append(sym); n -= v
    return "".join(s)

def grade_points_stakes(df_sorted: pd.DataFrame, Fs: float, Fh: float, loss_pct: float) -> np.ndarray:
    """
    Estacas que limitam as ondas (mudança de sinal do incremento CORRIGIDO).
    Corte (v>0): *Fs ; Aterro (v<0): *Fh*(1+perdas)
    """
    v = df_sorted["volume_m3"].to_numpy(dtype=float)
    v_corr = np.where(v >= 0, v * Fs, v * Fh)
    v_corr = np.where(v_corr < 0, v_corr * (1.0 + loss_pct/100.0), v_corr)

    x_est = pos_to_estaca(df_sorted["pos_m"].to_numpy())
    if len(v_corr) < 2:
        lo = float(np.nanmin(x_est)); hi = float(np.nanmax(x_est))
        return np.array([lo, hi], dtype=float)

    s = np.sign(v_corr)
    mids = list(np.where(s[:-1] * s[1:] < 0)[0] + 1)
    pts = [float(x_est[0])] + [float(x_est[i]) for i in mids] + [float(x_est[-1])]
    pts = sorted(set(round(p, 1) for p in pts))
    return np.array(pts, dtype=float)

def apply_block_ranges(df_sections: pd.DataFrame, block_ranges_est: List[Tuple[float, float]]) -> pd.DataFrame:
    """
    Define 'usar_em_aterro=False' para cortes cujo pos_m cai em QUALQUER faixa [Eini, Efim] (em estacas).
    """
    if "usar_em_aterro" not in df_sections.columns:
        df_sections["usar_em_aterro"] = True
    if not block_ranges_est:
        return df_sections
    est = pos_to_estaca(df_sections["pos_m"].to_numpy())
    mask_block = np.zeros(len(df_sections), dtype=bool)
    for a, b in block_ranges_est:
        a, b = (float(a), float(b))
        lo, hi = (min(a, b), max(a, b))
        mask_block |= (est >= lo) & (est <= hi)
    df_sections.loc[(mask_block) & (df_sections["volume_m3"] > 0), "usar_em_aterro"] = False
    return df_sections

# ============================================================
# Quadro (agregado por ondas / grade points)
# ============================================================
def build_distribution_table_waves(
    flows_intern: pd.DataFrame,
    flows_borrow: pd.DataFrame,
    flows_waste: pd.DataFrame,
    df_sorted: pd.DataFrame,
    Fs: float, Fh: float, loss_pct: float,
) -> pd.DataFrame:
    """
    Quadro agregado por ondas de compensação (entre grade points).
    DMT por linha = MT/Vol, com MT = ∑(V × dist_km).
    """
    wave_edges = grade_points_stakes(df_sorted, Fs, Fh, loss_pct)
    if len(wave_edges) < 2:
        wave_edges = np.array([0.0, 1.0], dtype=float)

    def wave_index(e: float) -> int:
        k = int(np.digitize([e], wave_edges, right=False)[0]) - 1
        return max(0, min(k, len(wave_edges) - 2))

    rows = []

    # 1) Internos
    if flows_intern is not None and not flows_intern.empty:
        tmp = []
        for _, r in flows_intern.iterrows():
            e0 = float(r["pos_corte_m"]) / 20.0
            e1 = float(r["pos_aterro_m"]) / 20.0
            w0, w1 = wave_index(min(e0, e1)), wave_index(max(e0, e1))
            w = w0 if w0 == w1 else wave_index(0.5 * (e0 + e1))
            tmp.append({
                "onda": w + 1,
                "e0": e0, "e1": e1,
                "vol": float(r["volume_m3"]),
                "mt": float(r["volume_m3"]) * float(r["dist_km"]),
            })
        dfi = pd.DataFrame(tmp)
        if not dfi.empty:
            g = dfi.groupby("onda", as_index=False)
            for _, gdf in g:
                onda = int(gdf["onda"].iloc[0])
                vol = float(gdf["vol"].sum())
                mt  = float(gdf["mt"].sum())
                dmt_km = (mt / vol) if vol > 1e-12 else 0.0
                o_da, o_a = int(np.floor(gdf["e0"].min())), int(np.ceil(gdf["e0"].max()))
                d_da, d_a = int(np.floor(gdf["e1"].min())), int(np.ceil(gdf["e1"].max()))
                rows.append({
                    "Onda": _roman(onda),
                    "Da estaca (O)": str(o_da),
                    "À estaca (O)":  str(o_a),
                    "Vol (m³)": round(vol, 0),
                    "Da estaca (D)": str(d_da),
                    "À estaca (D)":  str(d_a),
                    "DMT (km)": f"{dmt_km:.2f}",
                    "MT (m³·km)": f"{mt:.0f}",
                })

    # 2) Empréstimo → aterro
    if flows_borrow is not None and not flows_borrow.empty:
        tmp = []
        for _, r in flows_borrow.iterrows():
            ed = float(r["pos_aterro_m"]) / 20.0
            w = wave_index(ed)
            tmp.append({
                "onda": w + 1,
                "ed": ed,
                "vol": float(r["volume_m3"]),
                "mt":  float(r["volume_m3"]) * float(r["dist_km"]),
            })
        dfb = pd.DataFrame(tmp)
        if not dfb.empty:
            g = dfb.groupby("onda", as_index=False)
            for _, gdf in g:
                onda = int(gdf["onda"].iloc[0])
                vol = float(gdf["vol"].sum())
                mt  = float(gdf["mt"].sum())
                dmt_km = (mt / vol) if vol > 1e-12 else 0.0
                d_da, d_a = int(np.floor(gdf["ed"].min())), int(np.ceil(gdf["ed"].max()))
                rows.append({
                    "Onda": _roman(onda),
                    "Da estaca (O)": "EMPRÉSTIMO",
                    "À estaca (O)":  "",
                    "Vol (m³)": round(vol, 0),
                    "Da estaca (D)": str(d_da),
                    "À estaca (D)":  str(d_a),
                    "DMT (km)": f"{dmt_km:.2f}",
                    "MT (m³·km)": f"{mt:.0f}",
                })

    # 3) Corte → bota-fora
    if flows_waste is not None and not flows_waste.empty:
        tmp = []
        for _, r in flows_waste.iterrows():
            eo = float(r["pos_corte_m"]) / 20.0
            w = wave_index(eo)
            tmp.append({
                "onda": w + 1,
                "eo": eo,
                "vol": float(r["volume_m3"]),
                "mt":  float(r["volume_m3"]) * float(r["dist_km"]),
            })
        dfw = pd.DataFrame(tmp)
        if not dfw.empty:
            g = dfw.groupby("onda", as_index=False)
            for _, gdf in g:
                onda = int(gdf["onda"].iloc[0])
                vol = float(gdf["vol"].sum())
                mt  = float(gdf["mt"].sum())
                dmt_km = (mt / vol) if vol > 1e-12 else 0.0
                o_da, o_a = int(np.floor(gdf["eo"].min())), int(np.ceil(gdf["eo"].max()))
                rows.append({
                    "Onda": _roman(onda),
                    "Da estaca (O)": str(o_da),
                    "À estaca (O)":  str(o_a),
                    "Vol (m³)": round(vol, 0),
                    "Da estaca (D)": "BOTA-FORA",
                    "À estaca (D)":  "",
                    "DMT (km)": f"{dmt_km:.2f}",
                    "MT (m³·km)": f"{mt:.0f}",
                })

    tab = pd.DataFrame(rows, dtype="object")
    if tab.empty:
        return tab

    def _from_roman(s: str) -> int:
        m = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        val = 0; prev = 0
        for ch in s[::-1]:
            v = m.get(ch, 0)
            if v < prev: val -= v
            else: val += v; prev = v
        return val if val>0 else 10**9

    tab = tab.sort_values(by=["Onda"], key=lambda s: s.map(_from_roman)).reset_index(drop=True)

    # Evita erro Arrow: força colunas de estaca para texto
    text_cols = ["Da estaca (O)", "À estaca (O)", "Da estaca (D)", "À estaca (D)"]
    for c in text_cols:
        if c in tab.columns:
            tab[c] = tab[c].astype("string")

    return tab

# ============================================================
# Plot — Brückner (DMT em metros, sem linha/PP/setas)
# ============================================================
def plot_bruckner(
    df_sorted: pd.DataFrame,
    Fs: float, Fh: float, loss_factor: float,
    flows_intern: pd.DataFrame,
    flows_borrow: pd.DataFrame,
    flows_waste: pd.DataFrame,
    *, xtick_major: float = 20.0,
):
    import numpy as _np
    from matplotlib.lines import Line2D
    from collections import defaultdict

    # --- série acumulada ---
    v = df_sorted["volume_m3"].to_numpy(dtype=float)
    v_corr = _np.where(v >= 0, v * Fs, v * Fh * (1.0 + loss_factor / 100.0))
    y = _np.cumsum(v_corr)
    x_est = pos_to_estaca(df_sorted["pos_m"].to_numpy())

    # limites e escalas p/ lógica de rótulos
    xmin, xmax = float(_np.nanmin(x_est)), float(_np.nanmax(x_est))
    ymin, ymax = float(_np.nanmin(y)),    float(_np.nanmax(y))
    xr = max(1.0, xmax - xmin)
    yr = max(1.0, ymax - ymin)

    # largura típica de um intervalo
    span_est = infer_section_span_est(df_sorted)
    seg_w = max(0.6, span_est if span_est else xr / 40.0)

    fig, ax = plt.subplots()

    # curva base
    ax.plot(x_est, y, color="#808080", linewidth=1.2, marker="o", ms=3.5, alpha=0.7, zorder=2)

    ax.scatter(
        x_est, y,
        marker = "x",
        s = 30,
        linewidths=1.3,
        color = "black",
        zorder=8,
        clip_on=True,
    )

    # sobreposição grossa por trecho (verde = sobe/corte ; vermelho = desce/aterro)
    for i in range(len(x_est) - 1):
        x0, x1 = float(x_est[i]), float(x_est[i + 1])
        xs = _np.linspace(x0, x1, 24)
        ys = _np.interp(xs, x_est, y)
        color = "#2ecc71" if (y[i + 1] - y[i]) >= 0 else "#e74c3c"
        ax.plot(xs, ys, color=color, lw=5.0, alpha=0.8, zorder=3)

    # ---------- util: traçado ondulado (para marcar externo) ----------
    def wiggle_on_curve(center_e, color, amp_frac=0.018, freq=6, lw=2.8, alpha=0.95, z=4):
        a, b = max(xmin, center_e - seg_w/2), min(xmax, center_e + seg_w/2)
        if b <= a:  # nada a realçar
            return
        xs = _np.linspace(a, b, 120)
        base = _np.interp(xs, x_est, y)
        t = _np.linspace(0, 2*_np.pi*freq, xs.size)
        amp = amp_frac * yr
        ys = base + amp * _np.sin(t)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=z)

    # ---------- coletar eventos por estaca ----------
    def _roundx(e):  # arredonda p/ agrupar no mesmo ponto
        return float(_np.round(e, 6))

    cuts_at = defaultdict(list)     # x -> [(texto, y0)]
    fills_at = defaultdict(list)
    borrow_at = defaultdict(list)
    waste_at = defaultdict(list)

    # internos (pontos de CORTE e ATERRO)
    if flows_intern is not None and not flows_intern.empty:
        for _, r in flows_intern.iterrows():
            try:
                e0 = float(r["pos_corte_m"]) / 20.0
                e1 = float(r["pos_aterro_m"]) / 20.0
                vol = float(r["volume_m3"])
                dmt_m = float(r.get("dist_km", abs(e1 - e0) * 0.02)) * 1000.0
            except Exception:
                continue
            y0 = float(_np.interp(e0, x_est, y))
            y1 = float(_np.interp(e1, x_est, y))
            cuts_at[_roundx(e0)].append((f"CORTE\n{vol:.0f} m³\nDMT {dmt_m:.0f} m", y0))
            fills_at[_roundx(e1)].append((f"ATERRO\n{vol:.0f} m³\nDMT {dmt_m:.0f} m", y1))

    # externos
    if flows_borrow is not None and not flows_borrow.empty:
        for _, r in flows_borrow.iterrows():
            try:
                e1 = float(r["pos_aterro_m"]) / 20.0
                vol = float(r["volume_m3"])
                dkm = float(r.get("dist_km", 0.0))
            except Exception:
                continue
            y1 = float(_np.interp(e1, x_est, y))
            borrow_at[_roundx(e1)].append((f"EMPRÉSTIMO\n{vol:.0f} m³\nDMT {dkm*1000:.0f} m", y1))
            # marca o trecho como receptor de empréstimo
            wiggle_on_curve(e1, color="#2980b9")

    if flows_waste is not None and not flows_waste.empty:
        for _, r in flows_waste.iterrows():
            try:
                e0 = float(r["pos_corte_m"]) / 20.0
                vol = float(r["volume_m3"])
                dkm = float(r.get("dist_km", 0.0))
            except Exception:
                continue
            y0 = float(_np.interp(e0, x_est, y))
            waste_at[_roundx(e0)].append((f"BOTA-FORA\n{vol:.0f} m³\nDMT {dkm*1000:.0f} m", y0))
            # marca o trecho de onde o material sai para bota-fora
            wiggle_on_curve(e0, color="#f1c40f")

    # ---------- anotação inteligente (evita sobreposição / vira pro lado certo) ----------
    fs_anno = 4.5
    stack_level = defaultdict(int)  # (x, "up"/"down") -> nível
    alt_toggle = [True]             # alternador global

    def add_annot(x, y0, text, color, prefer=None, span=0.20):
        # orientação inicial
        if prefer == "up":
            sign = +1
        elif prefer == "down":
            sign = -1
        else:
            sign = +1 if alt_toggle[0] else -1
            alt_toggle[0] = not alt_toggle[0]

        # seta inclinada p/ dentro nos extremos
        dx = 0.0
        edge_tol = max(0.5, seg_w)
        if x - xmin < edge_tol:
            dx = +0.06 * xr
        elif xmax - x < edge_tol:
            dx = -0.06 * xr

        # tenta manter dentro dos limites em y
        base = span * yr
        ytxt = y0 + sign * base
        if ytxt > ymax - 0.05 * yr:
            sign = -1
        elif ytxt < ymin + 0.05 * yr:
            sign = +1
        base = span * yr

        # empilhamento por (x, orientação)
        key = (round(x, 3), "up" if sign > 0 else "down")
        lvl = stack_level[key]
        stack_level[key] += 1
        ytxt = y0 + sign * (base + lvl * 0.10 * yr)
        dx += (0.012 * xr) * (+1 if (lvl % 2 == 0) else -1)

        va = "bottom" if sign > 0 else "top"
        ax.annotate(
            text,
            xy=(x, y0),
            xytext=(x + dx, ytxt),
            ha="center",
            va=va,
            fontsize=fs_anno,
            color=color,
            textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.9, color=color),
            zorder=6,
            clip_on=False,
        )

    # regras de orientação quando há pares na mesma estaca
    all_xs = set(list(cuts_at.keys()) + list(fills_at.keys()) +
                 list(borrow_at.keys()) + list(waste_at.keys()))
    for x in sorted(all_xs):
        has_cut   = x in cuts_at
        has_fill  = x in fills_at
        has_borr  = x in borrow_at
        has_waste = x in waste_at

        # 1) corte & bota-fora no mesmo x: um pra cima, outro pra baixo
        if has_cut and has_waste:
            for text, y0 in cuts_at[x]:
                add_annot(x, y0, text, "#1e8449", prefer="up",   span=0.18)
            for text, y0 in waste_at[x]:
                add_annot(x, y0, text, "#c0392b", prefer="down", span=0.22)
        else:
            for text, y0 in cuts_at[x]:
                add_annot(x, y0, text, "#1e8449", span=0.18)
            for text, y0 in waste_at[x]:
                add_annot(x, y0, text, "#c0392b", span=0.22)

        # 2) aterro & empréstimo no mesmo x: um pra cima, outro pra baixo
        if has_fill and has_borr:
            for text, y0 in borrow_at[x]:
                add_annot(x, y0, text, "#1f4e79", prefer="up",   span=0.24)
            for text, y0 in fills_at[x]:
                add_annot(x, y0, text, "#922b21", prefer="down", span=0.18)
        else:
            for text, y0 in borrow_at[x]:
                add_annot(x, y0, text, "#1f4e79", span=0.24)
            for text, y0 in fills_at[x]:
                add_annot(x, y0, text, "#922b21", span=0.18)

    # --- eixos/estilo ---
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("Estaca"); ax.set_ylabel("Volume acumulado (m³)")
    ax.set_title("Diagrama de Brückner (Estacas)")

    if len(x_est) > 1:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin - 0.05*yr, ymax + 0.05*yr)  # um respiro p/ rótulos
        ax.xaxis.set_major_locator(MultipleLocator(xtick_major))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # legenda (dinâmica)
    legend_items = [
        Line2D([0],[0], color="#2ecc71", lw=6, alpha=0.85, label="Trecho em corte (↑)"),
        Line2D([0],[0], color="#e74c3c", lw=6, alpha=0.85, label="Trecho em aterro (↓)"),
    ]
    if flows_borrow is not None and not flows_borrow.empty:
        legend_items.append(Line2D([0],[0], color="#1f4e79", lw=3, label="Empréstimo"))
    if flows_waste is not None and not flows_waste.empty:
        legend_items.append(Line2D([0],[0], color="#aee41b", lw=3, label="Bota-fora"))

    fig.subplots_adjust(bottom=0.20 if len(legend_items) <= 3 else 0.26)
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18 if len(legend_items) <= 3 else -0.22),
        ncol=len(legend_items),
        frameon=False,
        borderaxespad=0.0,
        prop={"size": 5.0},  # ~50% do tamanho anterior
    )
    return fig




# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Terraplenagem — Momento de Transporte (LP)", layout="wide")
st.title("🚜 Terraplenagem — Momento de Transporte (LP com PuLP)")
st.caption("Fluxo: ① limitar cortes proibidos → ② equilibrar cortes/aterros internos ao máximo → ③ completar com empréstimo/bota-fora.")

# 0) Modo de entrada
mode = st.radio("Como você quer informar os dados?", ["Seções (corte + / aterro -)", "Ordenadas de Brückner (Y por estaca)"], index=0)

# 0.1) Entrada por seções
if mode == "Seções (corte + / aterro -)":
    with st.expander("📦 Exemplo rápido (10 seções a cada 100 m)", expanded=True):
        if st.button("Carregar exemplo de seções", key="btn_exemplo_secoes"):
            n = 10
            st.session_state["df"] = pd.DataFrame({
                "secao": list("ABCDEFGHIJ"),
                "pos_m": [i*100 for i in range(n)],
                "volume_m3": [200, -300, 400, -100, -200, -500, 600, -100, 300, -300],
                "usar_em_aterro": [True]*n,
            })
    df = st.session_state.get("df", pd.DataFrame({
        "secao": ["E0"] + [f"S{i:02d}" for i in range(1,10)],
        "pos_m": [0.0] + [i*100 for i in range(1,10)],
        "volume_m3": [0.0]*10,
        "usar_em_aterro": [True]*10,
    }))
    st.subheader("1) Dados dos segmentos (corte >0, aterro <0)")
    st.caption("Preencha a posição (m) e o volume (m³) de cada segmento. Volume positivo = corte; negativo = aterro.")
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    st.session_state["df"] = df

else:
    # 0.2) Entrada por ordenadas de Brückner
    st.subheader("1) Ordenadas de Brückner (Y por estaca)")
    st.caption("Informe as ordenadas acumuladas Y (m³) para cada estaca. O app converte ΔY em volumes de seções.")
    default_estacas = list(range(0, 211, 10))
    df_y = st.session_state.get("df_y", pd.DataFrame({"estaca": default_estacas, "Y_m3": [np.nan]*len(default_estacas)}))
    df_y = st.data_editor(df_y, use_container_width=True, num_rows="dynamic",
                          column_config={"estaca": st.column_config.NumberColumn("Estaca (20 m)")})
    st.session_state["df_y"] = df_y

    # Converter ordenadas em seções (inclui E0 com 0 m³)
    if st.button("Gerar seções a partir das ordenadas", key="btn_gerar_secoes"):
        y = df_y.dropna().sort_values("estaca").reset_index(drop=True)
        if len(y) < 2:
            st.error("Informe pelo menos duas ordenadas para gerar as seções.")
        else:
            e = y["estaca"].to_numpy(dtype=float)
            Y = y["Y_m3"].to_numpy(dtype=float)

            vol = np.diff(Y)                          # ΔY (m³) por intervalo
            e_ini = e[:-1]                            # estaca inicial do intervalo
            e_fim = e[1:]                             # estaca final
            pos_m = ((e_ini + e_fim) / 2.0) * 20.0    # posição no MEIO do intervalo (m)
            secao = [f"E{int(x)}" for x in e_ini]     # rótulo pela estaca inicial -> inclui E0

            d = pd.DataFrame({
                "secao": secao,
                "pos_m": pos_m,
                "volume_m3": vol,
                "usar_em_aterro": True
            })
            st.session_state["df"] = d
            st.success("Seções geradas a partir das ordenadas (ΔY) com E0 e posição no meio do intervalo.")

    # dataframe padrão das seções derivadas já com E0
    df = st.session_state.get("df", pd.DataFrame({
        "secao": [f"E{int(e)}" for e in default_estacas],
        "pos_m": [e*20.0 for e in default_estacas],
        "volume_m3": [0.0]*len(default_estacas),
        "usar_em_aterro": [True]*len(default_estacas),
    }))
    st.subheader("1b) Seções derivadas (edite se necessário)")
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    st.session_state["df"] = df

# 2) Sites externos
st.subheader("2) Jazidas de Empréstimo")
st.caption("Defina posição (m), capacidade (m³) e custo do material (R$/m³) de cada jazida.")
default_borrow = pd.DataFrame([{"nome":"E1","pos_m":500.0,"cap_m3":300.0,"c_material":0.0}])
borrow_sites = st.data_editor(st.session_state.get("borrow_sites", default_borrow),
                              use_container_width=True, num_rows="dynamic")
st.session_state["borrow_sites"] = borrow_sites

st.subheader("3) Bota-foras")
st.caption("Defina posição (m), capacidade (m³) e custo de descarte (R$/m³) de cada bota-fora.")
default_waste = pd.DataFrame([{"nome":"B1","pos_m":1200.0,"cap_m3":1000.0,"c_descarte":0.0}])
waste_sites = st.data_editor(st.session_state.get("waste_sites", default_waste),
                             use_container_width=True, num_rows="dynamic")
st.session_state["waste_sites"] = waste_sites

# --- helpers de saneamento ---
def _clean_borrow(df):
    df = df.copy()
    for c in ["pos_m", "cap_m3", "c_material"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["pos_m"])
    df["c_material"] = df["c_material"].fillna(0.0)
    df["cap_m3"]     = df["cap_m3"].fillna(1e12)  # evitar inf
    return df

def _clean_waste(df):
    df = df.copy()
    for c in ["pos_m", "cap_m3", "c_descarte"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["pos_m"])
    df["c_descarte"] = df["c_descarte"].fillna(0.0)
    df["cap_m3"]     = df["cap_m3"].fillna(1e12)
    return df

borrow_sites = _clean_borrow(borrow_sites)
waste_sites  = _clean_waste(waste_sites)
st.session_state["borrow_sites"] = borrow_sites
st.session_state["waste_sites"]  = waste_sites

# --------- TÓPICOS 4 e 5 LADO A LADO ----------
col4, col5 = st.columns(2)

with col4:
    st.subheader("4) Parâmetros financeiros e operacionais")
    st.caption("Fs (cortes) e Fh (aterros) ajustam os volumes; perdas afetam apenas o gráfico e o quadro por ondas.")
    c_transp_km = st.number_input("C_transporte (R$/m³·km)", min_value=0.0, value=1.0, step=0.1)
    Fh         = st.number_input("Fator de homogeneização/compactação Fh (aterros)", min_value=0.9, value=1.40, step=0.05)
    Fs         = st.number_input("Fator de empolamento Fs (cortes)", min_value=1.0, value=1.10, step=0.05)
    loss_perc  = st.number_input("Perdas [%] (apenas para gráfico/ondas)", min_value=0.0, value=5.0, step=1.0)

with col5:
    st.subheader("5) Restrições de uso do material (cortes proibidos para aterro)")
    st.caption("Informe faixas de estacas onde o material de corte NÃO pode ser usado em aterro.")
    apply_ex4 = st.checkbox("Aplicar automaticamente E40–E60 e E200–E210 (como no Exercício 4)", value=False)
    custom_ranges = st.text_input("Faixas adicionais (ex.: 0-10; 60-70)", value="")
    block_ranges = []
    if apply_ex4:
        block_ranges.extend([(40,60),(200,210)])
    if custom_ranges.strip():
        for token in custom_ranges.replace(";", ",").split(","):
            token = token.strip()
            if "-" in token:
                a,b = token.split("-",1)
                try:
                    block_ranges.append((float(a), float(b)))
                except Exception:
                    pass
df = apply_block_ranges(st.session_state["df"].copy(), block_ranges)

# --------- TÓPICOS 6 e 7 LADO A LADO ----------
col6, col7 = st.columns(2)

with col6:
    st.subheader("6) Opções do diagrama")
    st.caption("O eixo X é em estacas (1 estaca = 20 m). Padrão de marcação: a cada 20 estacas.")
    xtick_major = st.number_input("Espaçamento dos marcadores de estaca (eixo X)", min_value=1.0, value=20.0, step=1.0)

with col7:
    st.subheader("7) Distância MÁXIMA de movimentação interna")
    st.caption("Limite operacional: pares corte→aterro com distância maior que este valor ficam proibidos.")
    max_move_m = st.number_input(
        "Distância máxima permitida para movimentos corte→aterro internos (m)",
        min_value=0.0, value=2000.0, step=50.0,
        help="Pares corte→aterro com distância maior que este valor NÃO serão considerados no modelo."
    )

# botão centralizado e grande (com key única)
st.markdown("""
<style>
div.stButton > button:first-child {
    width: 100%;
    height: 3.2em;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)
sp1, sp2, sp3 = st.columns([1,2,1])
with sp2:
    clicked = st.button("Calcular as movimentações de terra mais eficientes", key="btn_calcular_main")

st.divider()

# ==== INÍCIO DO BLOCO DE RESOLUÇÃO ====
if clicked:
    if df.empty or "pos_m" not in df or "volume_m3" not in df:
        st.error("Tabela inválida. Inclua colunas 'pos_m' e 'volume_m3'.")
        st.stop()

    df_sorted = df.sort_values("pos_m").reset_index(drop=True)
    cuts  = df_sorted[df_sorted["volume_m3"] >  1e-9].reset_index(drop=True)
    fills = df_sorted[df_sorted["volume_m3"] < -1e-9].reset_index(drop=True)

    total_cut      = float(cuts["volume_m3"].sum())
    total_fill     = float(-fills["volume_m3"].sum())
    cap_borrow_tot = float(borrow_sites["cap_m3"].sum()) if not borrow_sites.empty else 0.0
    cap_waste_tot  = float(waste_sites["cap_m3"].sum())  if not waste_sites.empty  else float("inf")

    st.write(f"**Total corte:** {total_cut:.2f} m³ | **Total aterro:** {total_fill:.2f} m³")
    st.write(f"**Cap. total empréstimo:** {cap_borrow_tot:.2f} m³ | **Cap. total bota-fora:** {cap_waste_tot if np.isfinite(cap_waste_tot) else '∞'} m³")

    if total_cut + cap_borrow_tot + 1e-9 < total_fill:
        st.error("❌ Falta material: Corte + Empréstimo < Aterro.")
        st.stop()
    if total_cut - total_fill - cap_borrow_tot > cap_waste_tot + 1e-9:
        st.error("❌ Sobra de corte > capacidade total de bota-fora.")
        st.stop()

    I = list(cuts.index); J = list(fills.index)
    P = list(borrow_sites.index) if not borrow_sites.empty else []
    K = list(waste_sites.index)  if not waste_sites.empty  else []

    if "usar_em_aterro" not in cuts.columns:
        cuts["usar_em_aterro"] = True

    def dkm(a, b):
        return abs(float(a) - float(b)) / 1000.0

    # pares internos respeitando bloqueios e distância máxima
    intern_pairs = []
    for i in I:
        if not bool(cuts.loc[i, "usar_em_aterro"]):
            continue
        for j in J:
            if max_move_m > 0 and dkm(cuts.loc[i,"pos_m"], fills.loc[j,"pos_m"]) * 1000.0 > max_move_m + 1e-9:
                continue
            intern_pairs.append((i, j))
    if not intern_pairs and (I and J):
        st.warning("Nenhum par corte→aterro interno disponível (bloqueios e/ou distância máxima muito restritiva).")

    # ---------- FASE 1: MAXIMIZAR COMPENSAÇÃO INTERNA ----------
    prob1 = pulp.LpProblem("Phase1_Max_Internal", pulp.LpMaximize)
    x1 = pulp.LpVariable.dicts("x_intern", intern_pairs, lowBound=0, cat=pulp.LpContinuous)
    for i in I:
        prob1 += pulp.lpSum(x1[key] for key in x1 if key[0] == i) <= float(cuts.loc[i, "volume_m3"])
    for j in J:
        prob1 += pulp.lpSum(x1[key] for key in x1 if key[1] == j) <= float(-fills.loc[j, "volume_m3"])
    prob1 += pulp.lpSum(x1[key] for key in x1)
    status1 = prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    used_internal = {key: (x1[key].value() or 0.0) for key in x1}

    # ---------- FASE 2: COMPLETAR COM EMPRÉSTIMO/BOTA-FORA (MINIMIZAR CUSTO) ----------
    # custos apenas para chaves finitas
    cost_intern = {}
    for (i, j) in intern_pairs:
        dist = dkm(cuts.loc[i, "pos_m"], fills.loc[j, "pos_m"])
        if np.isfinite(dist):
            cost_intern[(i, j)] = c_transp_km * dist

    cost_borrow = {}
    for p in P:
        for j in J:
            pos_site = float(borrow_sites.loc[p,"pos_m"])
            pos_fill = float(fills.loc[j,"pos_m"])
            c_mat    = float(borrow_sites.loc[p,"c_material"])
            dist     = dkm(pos_site, pos_fill)
            if all(np.isfinite([pos_site, pos_fill, c_mat, dist])):
                cost_borrow[(p,j)] = c_transp_km * dist + c_mat

    cost_waste = {}
    for i in I:
        for k in K:
            pos_cut  = float(cuts.loc[i,"pos_m"])
            pos_wast = float(waste_sites.loc[k,"pos_m"])
            c_disp   = float(waste_sites.loc[k,"c_descarte"])
            dist     = dkm(pos_cut, pos_wast)
            if all(np.isfinite([pos_cut, pos_wast, c_disp, dist])):
                cost_waste[(i,k)] = c_transp_km * dist + c_disp

    x2 = pulp.LpVariable.dicts("x_intern_fix", list(cost_intern.keys()), lowBound=0, cat=pulp.LpContinuous)
    b  = pulp.LpVariable.dicts("b_borrow",    list(cost_borrow.keys()), lowBound=0, cat=pulp.LpContinuous) if cost_borrow else {}
    w  = pulp.LpVariable.dicts("w_waste",     list(cost_waste.keys()),  lowBound=0, cat=pulp.LpContinuous) if cost_waste  else {}

    prob2 = pulp.LpProblem("Phase2_Min_Cost", pulp.LpMinimize)
    prob2 += (
        pulp.lpSum(cost_intern[k] * x2[k] for k in cost_intern) +
        pulp.lpSum(cost_borrow[k] * b[k]  for k in cost_borrow) +
        pulp.lpSum(cost_waste[k]  * w[k]  for k in cost_waste)
    )

    # fixa o que saiu da Fase 1
    for (i, j) in intern_pairs:
        if (i, j) in x2:
            val = used_internal.get((i, j), 0.0) or 0.0
            x2[(i, j)].lowBound = val
            x2[(i, j)].upBound  = val

    # balanço
    for i in I:
        lhs_x = pulp.lpSum(x2[k] for k in x2 if k[0] == i)
        lhs_w = pulp.lpSum(w[k]  for k in w  if k[0] == i) if w else 0
        prob2 += lhs_x + lhs_w == float(cuts.loc[i, "volume_m3"])

    for j in J:
        lhs_x = pulp.lpSum(x2[k] for k in x2 if k[1] == j)
        lhs_b = pulp.lpSum(b[k]  for k in b  if k[1] == j) if b else 0
        prob2 += lhs_x + lhs_b == float(-fills.loc[j, "volume_m3"])

    for p in P:
        prob2 += pulp.lpSum(b[k] for k in b if k[0] == p) <= float(borrow_sites.loc[p, "cap_m3"])
    for k_ in K:
        prob2 += pulp.lpSum(w[k] for k in w if k[1] == k_) <= float(waste_sites.loc[k_, "cap_m3"])

    status2 = prob2.solve(pulp.PULP_CBC_CMD(msg=False))

    # explicação dos status
    st.info(
        "• **Fase 1 (interno)**: maximiza o volume movimentado **apenas** entre cortes e aterros no traçado, "
        "respeitando faixas proibidas e a distância máxima.\n"
        "• **Fase 2 (completar)**: com esses volumes **fixados**, minimiza o **custo total** usando "
        "empréstimo e/ou bota-fora (com capacidades/custos).\n"
        "Status: ‘Optimal’ = ótima; ‘Feasible’ = viável (não necessariamente ótima); ‘Infeasible’ = sem solução."
    )
    st.write(f"**Status Fase 1 (interno):** {pulp.LpStatus[status1]}  |  **Status Fase 2 (completar):** {pulp.LpStatus[status2]}")

    if pulp.LpStatus[status2] not in ("Optimal", "Feasible"):
        st.error("Modelo não encontrou solução viável na Fase 2.")
        st.stop()

    # ---------- EXTRAIR FLUXOS ----------
    flows_intern = pd.DataFrame([
        {
            "corte_secao": cuts.loc[i, "secao"],
            "aterro_secao": fills.loc[j, "secao"],
            "pos_corte_m": float(cuts.loc[i, "pos_m"]),
            "pos_aterro_m": float(fills.loc[j, "pos_m"]),
            "dist_km": dkm(cuts.loc[i, "pos_m"], fills.loc[j, "pos_m"]),
            "volume_m3": (x2[(i, j)].value() or 0.0),
        }
        for (i, j) in x2.keys()
        if (x2[(i, j)].value() or 0.0) > 1e-8
    ]).reset_index(drop=True)

    flows_borrow = pd.DataFrame([
        {
            "site": borrow_sites.loc[p, "nome"],
            "pos_site_m": float(borrow_sites.loc[p, "pos_m"]),
            "aterro_secao": fills.loc[j, "secao"],
            "pos_aterro_m": float(fills.loc[j, "pos_m"]),
            "dist_km": dkm(borrow_sites.loc[p, "pos_m"], fills.loc[j, "pos_m"]),
            "volume_m3": (b[(p, j)].value() or 0.0),
        }
        for (p, j) in (b.keys() if b else [])
        if (b[(p, j)].value() or 0.0) > 1e-8
    ]).reset_index(drop=True)

    flows_waste = pd.DataFrame([
        {
            "site": waste_sites.loc[k, "nome"],
            "pos_site_m": float(waste_sites.loc[k, "pos_m"]),
            "corte_secao": cuts.loc[i, "secao"],
            "pos_corte_m": float(cuts.loc[i, "pos_m"]),
            "dist_km": dkm(cuts.loc[i, "pos_m"], waste_sites.loc[k, "pos_m"]),
            "volume_m3": (w[(i, k)].value() or 0.0),
        }
        for (i, k) in (w.keys() if w else [])
        if (w[(i, k)].value() or 0.0) > 1e-8
    ]).reset_index(drop=True)

    # ---------- QUADRO ----------
    st.subheader("Quadro de Distribuição de Terras (por ondas)")
    st.caption("Cada linha agrega os movimentos dentro de uma ‘onda’ (entre pontos de passagem). **DMT (km)** = **MT/Vol** (com MT em m³·km).")
    tab = build_distribution_table_waves(flows_intern, flows_borrow, flows_waste, df_sorted, Fs, Fh, loss_perc)
    if tab.empty:
        st.info("Sem movimentos registrados.")
    else:
        cols_order = ["Onda","Da estaca (O)","À estaca (O)","Vol (m³)","Da estaca (D)","À estaca (D)","DMT (km)","MT (m³·km)"]
        tab = tab[[c for c in cols_order if c in tab.columns]]
        st.dataframe(tab, use_container_width=True)
        st.download_button("⬇️ Baixar quadro (.csv)", data=tab.to_csv(index=False).encode("utf-8"),
                           file_name="quadro_distribuicao_terras.csv", mime="text/csv")

    # ---------- CUSTO TOTAL ----------
    total_cost = float(pulp.value(prob2.objective))
    st.success(f"**Custo total estimado** = {total_cost:,.2f}  (pondera MT pelos custos/capacidades de sites)")

    # ---------- TABELAS AUXILIARES ----------
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Compensação longitudinal (corte→aterro) — utilizados")
        st.dataframe(flows_intern if not flows_intern.empty else pd.DataFrame([{"info":"—"}]), use_container_width=True)
    with c2:
        st.markdown("### Externos (empréstimo / bota-fora)")
        if flows_borrow.empty and flows_waste.empty:
            st.info("Sem movimentos externos.")
        else:
            if not flows_borrow.empty:
                st.write("**Empréstimo → aterro**")
                st.dataframe(flows_borrow, use_container_width=True)
            if not flows_waste.empty:
                st.write("**Corte → bota-fora**")
                st.dataframe(flows_waste, use_container_width=True)

    # ---------- GRÁFICO ----------
    st.subheader("Gráfico — Diagrama de Brückner (Estacas)")
    st.caption("Anotações mostram **DMT em metros** para cada movimento destacado.")
    st.pyplot(
        plot_bruckner(
            df_sorted, Fs, Fh, loss_perc,
            flows_intern=flows_intern, flows_borrow=flows_borrow, flows_waste=flows_waste,
            xtick_major=xtick_major
        )
    )

    # ---------- DOWNLOAD JSON ----------
    result = {
        "flows_intern": flows_intern.to_dict(orient="records"),
        "flows_borrow": flows_borrow.to_dict(orient="records"),
        "flows_waste":  flows_waste.to_dict(orient="records"),
        "objective_cost": total_cost,
        "params": {
            "c_transporte_R$/m3.km": c_transp_km, "Fh": Fh, "Fs": Fs, "loss_perc_%": loss_perc,
            "borrow_sites": borrow_sites.to_dict(orient="records"),
            "waste_sites":  waste_sites.to_dict(orient="records"),
            "block_ranges_est": block_ranges, "max_move_m": max_move_m,
        }
    }
    st.download_button("⬇️ Baixar resultados (.json)",
                       data=json.dumps(result, indent=2, ensure_ascii=False),
                       file_name="resultado_terraplenagem.json")
# ==== FIM DO BLOCO DE RESOLUÇÃO ====
else:
    st.info("Preencha os dados e clique em **Calcular as movimentações de terra mais eficientes**.")
