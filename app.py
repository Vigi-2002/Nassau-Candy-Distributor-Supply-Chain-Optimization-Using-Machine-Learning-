"""
Nassau Candy Distributor — Factory Optimization Streamlit App
=============================================================
Run locally:
    pip install streamlit plotly pandas numpy
    streamlit run app.py

Place all CSV/PKL outputs from the pipeline in a 'data/' folder
next to this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Factory Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour constants ───────────────────────────────────────────────────────────
CLR_BLUE   = "#378ADD"
CLR_GREEN  = "#1D9E75"
CLR_AMBER  = "#EF9F27"
CLR_RED    = "#D85A30"
CLR_PURPLE = "#7F77DD"
CLR_GRAY   = "#888780"

RISK_CLR  = {"Low": CLR_GREEN,  "Medium": CLR_AMBER, "High": CLR_RED}
DIV_CLR   = {"Chocolate": CLR_BLUE, "Sugar": CLR_AMBER, "Other": CLR_GREEN}
FAC_CLR   = {
    "Lot's O' Nuts":     CLR_BLUE,
    "Wicked Choccy's":  CLR_GREEN,
    "Sugar Shack":       CLR_AMBER,
    "Secret Factory":    CLR_PURPLE,
    "The Other Factory": CLR_RED,
}

PROD_NAMES = {
    'CHO-FUD-51000': 'Fudge Mallows',       'CHO-MIL-31000': 'Milk Chocolate',
    'CHO-NUT-13000': 'Nutty Crunch',        'CHO-SCR-58000': 'Scrumdiddlyumptious',
    'CHO-TRI-54000': 'Triple Dazzle',       'OTH-FIZ-56000': 'Fizzy Lifting',
    'OTH-GUM-21000': 'Wonka Gum',           'OTH-KAZ-38000': 'Kazookles',
    'OTH-LIC-15000': 'Lickable Wallpaper',  'SUG-FUN-75000': 'Fun Dip',
    'SUG-HAI-55000': 'Hair Toffee',         'SUG-LAF-25000': 'Laffy Taffy',
    'SUG-NER-92000': 'Nerds',               'SUG-SWE-91000': 'SweeTARTS',
}

# ── Data loader ────────────────────────────────────────────────────────────────
DATA_DIR = "outputs"

@st.cache_data
def load_data():
    enrich      = pd.read_csv(f"{DATA_DIR}/nassau_enriched.csv")
    sim         = pd.read_csv(f"{DATA_DIR}/stage4_simulations.csv")
    recs        = pd.read_csv(f"{DATA_DIR}/stage5_recommendations.csv")
    top_recs    = pd.read_csv(f"{DATA_DIR}/stage5_top_recommendations.csv")
    prod_cl     = pd.read_csv(f"{DATA_DIR}/stage3_product_clusters.csv")
    route_cl    = pd.read_csv(f"{DATA_DIR}/stage3_route_clusters.csv")
    enrich['Product Name'] = enrich['Product ID'].map(PROD_NAMES)
    return enrich, sim, recs, top_recs, prod_cl, route_cl

enrich, sim, recs, top_recs, prod_cl, route_cl = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍬 Nassau Candy")
    st.caption("Factory Optimization Suite")
    st.divider()

    st.subheader("Filters")

    all_products = {v: k for k, v in PROD_NAMES.items()}
    sel_product  = st.selectbox(
        "Product",
        options=list(PROD_NAMES.values()),
        index=0,
    )
    sel_pid = all_products[sel_product]

    sel_regions = st.multiselect(
        "Region",
        options=sorted(enrich['Region'].unique()),
        default=sorted(enrich['Region'].unique()),
    )

    sel_modes = st.multiselect(
        "Ship mode",
        options=sorted(enrich['Ship Mode'].unique()),
        default=sorted(enrich['Ship Mode'].unique()),
    )

    st.divider()
    st.subheader("Optimization Priority")
    priority = st.slider(
        "Speed  ◄──────►  Profit",
        min_value=0, max_value=100, value=50,
        help="Slide left to prioritize lead time reduction, right to prioritize profit impact."
    )
    w_lt     = round((100 - priority) / 100, 2)
    w_profit = round(priority / 100, 2)
    w_conf   = 0.25  # fixed
    # Renormalize LT + Profit weights to leave room for confidence
    scale    = 0.75 / (w_lt + w_profit) if (w_lt + w_profit) > 0 else 1
    w_lt    *= scale
    w_profit*= scale

    st.caption(f"Weights → LT: {w_lt:.2f} · Profit: {w_profit:.2f} · Confidence: {w_conf:.2f}")
    st.divider()
    st.caption("Stages 1–5 pipeline | Nassau Candy internship project")

# ── Apply filters to base data ─────────────────────────────────────────────────
filtered = enrich[
    (enrich['Region'].isin(sel_regions)) &
    (enrich['Ship Mode'].isin(sel_modes))
].copy()

prod_filtered = filtered[filtered['Product ID'] == sel_pid]

# ── Recompute recommendation scores with current slider weights ────────────────
def recompute_scores(df_sim, wlt, wpr, wco):
    v = df_sim[df_sim['LT Reduction (%)'] > 0].copy()
    def minmax(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else pd.Series(np.zeros(len(s)), index=s.index)
    v['LT_norm']     = minmax(v['LT Reduction (%)'])
    v['Profit_norm'] = minmax(v['Profit Impact ($)'])
    v['Conf_norm']   = v['Confidence Score']
    RISK_PENALTY = {'Low': 0.00, 'Medium': 0.05, 'High': 0.20}
    v['Score'] = (wlt * v['LT_norm'] + wpr * v['Profit_norm'] +
                  wco * v['Conf_norm'] - v['Risk'].map(RISK_PENALTY))
    return v.sort_values('Score', ascending=False).reset_index(drop=True)

scored = recompute_scores(sim, w_lt, w_profit, w_conf)

# ══════════════════════════════════════════════════════════════════════════════
# Navigation tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭  Factory Optimizer",
    "🔄  What-If Scenario",
    "🏆  Recommendations",
    "⚠️  Risk & Impact",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FACTORY OPTIMIZATION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Factory Optimization Simulator")
    st.caption(f"Showing performance for **{sel_product}** across all factories · "
               f"Regions: {', '.join(sel_regions)} · Ship modes: {', '.join(sel_modes)}")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    prod_info   = prod_cl[prod_cl['Product ID'] == sel_pid].iloc[0] if len(prod_cl[prod_cl['Product ID'] == sel_pid]) else None
    prod_orders = enrich[enrich['Product ID'] == sel_pid]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current factory",
                  prod_orders['Factory'].iloc[0] if len(prod_orders) else "—")
    with col2:
        st.metric("Avg lead time",
                  f"{prod_orders['Lead_Time'].mean():.0f} days" if len(prod_orders) else "—")
    with col3:
        st.metric("Avg gross profit",
                  f"${prod_orders['Gross Profit'].mean():.2f}" if len(prod_orders) else "—")
    with col4:
        st.metric("Performance cluster",
                  prod_info['Cluster_Label'] if prod_info is not None else "—")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Lead time by factory (simulated) ──────────────────────────────────────
    with col_left:
        st.subheader("Predicted lead time by factory")
        prod_sim = sim[sim['Product ID'] == sel_pid].copy()
        current_fac = prod_orders['Factory'].iloc[0] if len(prod_orders) else None
        current_lt  = prod_orders['Lead_Time'].mean() if len(prod_orders) else 0

        # Build comparison table: current + alternates
        rows = [{"Factory": current_fac, "Avg LT (days)": round(current_lt, 1),
                 "Type": "Current", "LT Reduction (%)": 0}]
        for _, r in prod_sim.iterrows():
            rows.append({
                "Factory": r['Alternate Factory'],
                "Avg LT (days)": round(r['Simulated Avg LT (days)'], 1),
                "Type": "Alternate",
                "LT Reduction (%)": round(r['LT Reduction (%)'], 2),
            })
        fac_df = pd.DataFrame(rows).sort_values("Avg LT (days)")

        fig = px.bar(
            fac_df, x="Avg LT (days)", y="Factory", orientation='h',
            color="Type",
            color_discrete_map={"Current": CLR_GRAY, "Alternate": CLR_BLUE},
            text="Avg LT (days)",
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=280, legend_title_text="",
            xaxis_title="Avg lead time (days)",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Shipping distance by factory ──────────────────────────────────────────
    with col_right:
        st.subheader("Shipping distance by factory")
        dist_rows = [{"Factory": current_fac,
                      "Avg Distance (mi)": round(prod_orders['Shipping_Distance_Miles'].mean(), 0),
                      "Type": "Current"}]
        for _, r in prod_sim.iterrows():
            dist_rows.append({
                "Factory": r['Alternate Factory'],
                "Avg Distance (mi)": round(r['Simulated Avg Dist (mi)'], 0),
                "Type": "Alternate",
            })
        dist_df = pd.DataFrame(dist_rows).sort_values("Avg Distance (mi)")

        fig2 = px.bar(
            dist_df, x="Avg Distance (mi)", y="Factory", orientation='h',
            color="Type",
            color_discrete_map={"Current": CLR_GRAY, "Alternate": CLR_GREEN},
            text="Avg Distance (mi)",
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=280, legend_title_text="",
            xaxis_title="Avg shipping distance (miles)",
            yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Route cluster heatmap ─────────────────────────────────────────────────
    st.subheader("Lead time heatmap — Factory × Region")
    heat_data = route_cl.pivot_table(
        index='Factory', columns='Region',
        values='Avg_Lead_Time', aggfunc='mean'
    ).round(0)
    fig3 = px.imshow(
        heat_data,
        color_continuous_scale='RdYlGn_r',
        text_auto=True,
        aspect='auto',
        labels=dict(color="Avg LT (days)"),
    )
    fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WHAT-IF SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("What-If Scenario Analysis")
    st.caption(f"Compare current assignment vs alternate factory for **{sel_product}**")

    prod_sim = sim[sim['Product ID'] == sel_pid].copy()
    if len(prod_sim) == 0:
        st.warning("No simulation data for this product.")
    else:
        alt_factories = prod_sim['Alternate Factory'].unique().tolist()
        sel_alt = st.selectbox("Choose alternate factory to compare", options=alt_factories)

        row = prod_sim[prod_sim['Alternate Factory'] == sel_alt].iloc[0]
        current_lt  = row['Current Avg LT (days)']
        sim_lt      = row['Simulated Avg LT (days)']
        lt_delta    = row['LT Reduction (days)']
        lt_pct      = row['LT Reduction (%)']
        dist_delta  = row['Distance Delta (mi)']
        profit_imp  = row['Profit Impact ($)']
        confidence  = row['Confidence Score']
        risk        = row['Risk']

        # ── Comparison metric cards ────────────────────────────────────────
        st.subheader("Scenario comparison")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current lead time",  f"{current_lt:.0f} days")
        c2.metric("Simulated lead time", f"{sim_lt:.0f} days",
                  delta=f"{-lt_delta:.0f} days",
                  delta_color="inverse")
        c3.metric("LT reduction", f"{lt_pct:.1f}%",
                  delta_color="normal" if lt_pct > 0 else "inverse")
        c4.metric("Profit impact",  f"${profit_imp:.0f}")
        c5.metric("Confidence",     f"{confidence:.2f}")

        st.divider()
        col_l, col_r = st.columns(2)

        # ── Side-by-side LT bar ────────────────────────────────────────────
        with col_l:
            st.subheader("Lead time — current vs simulated")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=["Current", "Simulated"],
                y=[current_lt, sim_lt],
                marker_color=[CLR_GRAY, CLR_GREEN if lt_pct > 0 else CLR_RED],
                text=[f"{current_lt:.0f}d", f"{sim_lt:.0f}d"],
                textposition='outside',
            ))
            fig_comp.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="Lead time (days)",
                yaxis_range=[min(current_lt, sim_lt)*0.92,
                             max(current_lt, sim_lt)*1.06],
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # ── Distance delta ─────────────────────────────────────────────────
        with col_r:
            st.subheader("Shipping distance change")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=["Current", "Simulated"],
                y=[row['Current Avg Dist (mi)'], row['Simulated Avg Dist (mi)']],
                marker_color=[CLR_GRAY, CLR_GREEN if dist_delta > 0 else CLR_RED],
                text=[f"{row['Current Avg Dist (mi)']:.0f} mi",
                      f"{row['Simulated Avg Dist (mi)']:.0f} mi"],
                textposition='outside',
            ))
            fig_dist.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="Avg distance (miles)",
                yaxis_range=[0, max(row['Current Avg Dist (mi)'],
                                   row['Simulated Avg Dist (mi)']) * 1.15],
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # ── All alternate factories comparison ─────────────────────────────
        st.subheader(f"All alternate factories — {sel_product}")
        all_alts = prod_sim[['Alternate Factory', 'LT Reduction (%)',
                              'LT Reduction (days)', 'Distance Delta (mi)',
                              'Profit Impact ($)', 'Confidence Score',
                              'Risk']].copy()
        all_alts = all_alts.sort_values('LT Reduction (%)', ascending=False)

        fig_all = px.bar(
            all_alts, x='Alternate Factory', y='LT Reduction (%)',
            color='Risk',
            color_discrete_map=RISK_CLR,
            text='LT Reduction (%)',
            title="Lead time reduction (%) by alternate factory",
        )
        fig_all.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_all.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_all.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                              xaxis_title="", yaxis_title="LT reduction (%)")
        st.plotly_chart(fig_all, use_container_width=True)

        # ── Risk badge ─────────────────────────────────────────────────────
        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.info(f"{risk_emoji.get(risk,'⚪')} Scenario risk level: **{risk}**  |  "
                f"Route cluster: **{row['Current Route Cluster']}** → "
                f"**{row['Alternate Route Cluster']}**  |  "
                f"Order count used for simulation: **{int(row['Order Count'])}**")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RECOMMENDATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Recommendation Dashboard")
    st.caption("Ranked factory reassignment suggestions based on current optimization priority")

    # ── KPI summary strip ─────────────────────────────────────────────────────
    low_risk = scored[scored['Risk'] == 'Low']
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg LT reduction",
              f"{low_risk['LT Reduction (days)'].mean():.1f} days",
              f"{low_risk['LT Reduction (%)'].mean():.1f}%")
    k2.metric("Avg profit impact",
              f"${low_risk['Profit Impact ($)'].mean():.0f}",
              "per product")
    k3.metric("Avg confidence", f"{low_risk['Confidence Score'].mean():.2f}")
    k4.metric("Coverage",
              f"{scored[scored['Risk']!='High']['Product ID'].nunique()}"
              f"/{sim['Product ID'].nunique()} products")

    st.divider()

    # ── Priority-adjusted scored table ────────────────────────────────────────
    st.subheader("All viable scenarios — ranked by current priority")
    display_cols = ['Product Name', 'Current Factory', 'Alternate Factory',
                    'LT Reduction (%)', 'LT Reduction (days)',
                    'Profit Impact ($)', 'Confidence Score', 'Score', 'Risk']
    display_df = scored[display_cols].head(20).copy()
    display_df['LT Reduction (%)']  = display_df['LT Reduction (%)'].round(2)
    display_df['LT Reduction (days)'] = display_df['LT Reduction (days)'].round(1)
    display_df['Profit Impact ($)']  = display_df['Profit Impact ($)'].round(0)
    display_df['Score']              = display_df['Score'].round(4)

    def colour_risk(val):
        c = {"Low": "#EAF3DE", "Medium": "#FAEEDA", "High": "#FCEBEB"}.get(val, "")
        return f"background-color: {c}"

    st.dataframe(
        display_df.style.applymap(colour_risk, subset=['Risk']),
        use_container_width=True, height=380,
    )

    st.divider()
    col_l2, col_r2 = st.columns(2)

    # ── Recommendation score chart ─────────────────────────────────────────────
    with col_l2:
        st.subheader("Score breakdown — top 10")
        top10 = scored.head(10).copy()
        fig_score = go.Figure()
        for comp, col, label in [
            ('LT_norm', CLR_BLUE, f'LT reduction ({int(w_lt*100)}%)'),
            ('Profit_norm', CLR_GREEN, f'Profit impact ({int(w_profit*100)}%)'),
            ('Conf_norm', CLR_AMBER, f'Confidence ({int(w_conf*100)}%)'),
        ]:
            weight = {'LT_norm': w_lt, 'Profit_norm': w_profit, 'Conf_norm': w_conf}[comp]
            fig_score.add_trace(go.Bar(
                y=top10['Product Name'] + ' → ' + top10['Alternate Factory'],
                x=top10[comp] * weight,
                orientation='h', name=label,
                marker_color=col,
            ))
        fig_score.update_layout(
            barmode='stack', height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Score contribution",
            yaxis_title="", legend_title_text="",
        )
        st.plotly_chart(fig_score, use_container_width=True)

    # ── Scatter: LT reduction vs profit ───────────────────────────────────────
    with col_r2:
        st.subheader("LT reduction vs profit impact")
        # Plotly size must be >= 0 — clip negative scores caused by risk penalties
        scatter_df = scored.copy()
        scatter_df['Bubble Size'] = scatter_df['Score'].clip(lower=0)
        fig_scatter = px.scatter(
            scatter_df,
            x='LT Reduction (%)',
            y='Profit Impact ($)',
            color='Risk',
            size='Bubble Size',
            size_max=22,
            hover_data=['Product Name', 'Alternate Factory', 'Confidence Score'],
            color_discrete_map=RISK_CLR,
        )
        fig_scatter.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
        fig_scatter.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1)
        fig_scatter.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Lead time reduction (%)",
            yaxis_title="Profit impact ($)",
            legend_title_text="Risk",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK & IMPACT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Risk & Impact Panel")
    st.caption("Profit impact alerts and high-risk reassignment warnings")

    # ── High-risk warnings ────────────────────────────────────────────────────
    high_risk = scored[scored['Risk'] == 'High']
    if len(high_risk):
        st.subheader("🔴 High-risk reassignment warnings")
        for _, r in high_risk.head(6).iterrows():
            reason = "Low order volume (< 10 orders)" if r['Order Count'] < 10 \
                     else "Lead time increases with this reassignment" if r['LT Reduction (%)'] < 0 \
                     else "Low confidence score (< 0.40)"
            st.warning(
                f"**{r['Product Name']}** → {r['Alternate Factory']}  |  "
                f"LT Δ: {r['LT Reduction (%)']:.1f}%  |  "
                f"Confidence: {r['Confidence Score']:.2f}  |  "
                f"⚠ {reason}"
            )
    else:
        st.success("No high-risk scenarios in the current filtered view.")

    st.divider()
    col_r1, col_r2 = st.columns(2)

    # ── Profit impact by product ───────────────────────────────────────────────
    with col_r1:
        st.subheader("Profit impact — all viable scenarios")
        profit_df = (scored.groupby(['Product Name', 'Risk'])['Profit Impact ($)']
                           .max().reset_index()
                           .sort_values('Profit Impact ($)', ascending=True))
        fig_profit = px.bar(
            profit_df,
            x='Profit Impact ($)', y='Product Name',
            color='Risk', orientation='h',
            color_discrete_map=RISK_CLR,
        )
        fig_profit.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1)
        fig_profit.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Max profit impact ($)",
            yaxis_title="", legend_title_text="Risk",
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    # ── Confidence vs risk ─────────────────────────────────────────────────────
    with col_r2:
        st.subheader("Confidence score distribution by risk level")
        fig_box = px.box(
            scored, x='Risk', y='Confidence Score',
            color='Risk', color_discrete_map=RISK_CLR,
            category_orders={'Risk': ['Low', 'Medium', 'High']},
            points='all',
        )
        fig_box.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Risk level",
            yaxis_title="Confidence score",
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ── Factory workload impact ────────────────────────────────────────────────
    st.subheader("Factory workload — if top recommendations are actioned")
    current_load = enrich['Factory'].value_counts().reset_index()
    current_load.columns = ['Factory', 'Current Orders']

    best_per_prod = scored.groupby('Product ID').first().reset_index()
    order_delta = {}
    for _, r in best_per_prod.iterrows():
        d = r['Order Count']
        order_delta[r['Current Factory']]  = order_delta.get(r['Current Factory'], 0) - d
        order_delta[r['Alternate Factory']] = order_delta.get(r['Alternate Factory'], 0) + d

    current_load['Order Delta']      = current_load['Factory'].map(order_delta).fillna(0).astype(int)
    current_load['Projected Orders'] = current_load['Current Orders'] + current_load['Order Delta']
    current_load['Load Change (%)']  = (
        current_load['Order Delta'] / current_load['Current Orders'] * 100
    ).round(1)

    fig_wl = go.Figure()
    fig_wl.add_trace(go.Bar(
        name='Current', x=current_load['Factory'],
        y=current_load['Current Orders'],
        marker_color=CLR_GRAY, opacity=0.6,
    ))
    fig_wl.add_trace(go.Bar(
        name='Projected', x=current_load['Factory'],
        y=current_load['Projected Orders'],
        marker_color=[CLR_GREEN if d < 0 else CLR_AMBER if d == 0 else CLR_RED
                      for d in current_load['Order Delta']],
    ))
    fig_wl.update_layout(
        barmode='group', height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Order count",
        xaxis_title="",
        legend_title_text="",
    )
    st.plotly_chart(fig_wl, use_container_width=True)

    # ── Capacity warning ──────────────────────────────────────────────────────
    overloaded = current_load[current_load['Load Change (%)'] > 200]
    if len(overloaded):
        for _, r in overloaded.iterrows():
            st.error(
                f"⚠️ **Capacity alert — {r['Factory']}**: projected order load increases "
                f"by **{r['Load Change (%)']:.0f}%** ({r['Current Orders']:,} → "
                f"{r['Projected Orders']:,} orders). "
                f"Recommend phased implementation."
            )

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("📋  View full simulation data"):
        st.dataframe(sim, use_container_width=True)