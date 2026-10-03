"""Streamlit demonstrator for QSense."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.physics import TAUS_US, omega_from_b, b_from_omega, probability_zero
from core.sim import Noise, simulate
from core.crb import local_sigma
from est.bayes import summary as bayes_summary
try:
    from est.nn_infer import summary as nn_summary, load_weights
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False


st.set_page_config(page_title="QSense", layout="wide")
st.title("QSense · Learned Quantum Magnetometry")
st.caption("Testing Bayesian vs Amortized estimators under noise mismatch.")

tab_single, tab_bench = st.tabs(["1. Run & Compare", "2. Benchmark"])

with st.sidebar:
    st.header("Sensor and experiment")
    field = st.slider("True magnetic field (µT)", 0.0, 10.0, 4.0, .05)
    t2 = st.slider("True T2* (µs)", 2.5, 40.0, 10.0, .5)
    shape = st.slider("Decay shape", 1.0, 3.0, 1.0, .1)
    visibility = st.slider("Visibility", .4, .98, .9, .01)
    drift = st.slider("Phase drift σ at 16 µs (rad)", 0.0, 3.0, 0.0, .1)
    repeats = st.select_slider("Repeats (k)", options=[1, 2, 4, 8, 16], value=4)
    seed = st.number_input("Random seed", min_value=0, value=42)

omega = float(omega_from_b(field))
noise = Noise(t2_us=t2, shape=shape, visibility=visibility, sigma_phi_rad=drift)

with tab_single:
    st.subheader("1. Ramsey Fringe")
    tau_fine = np.linspace(0.01, 16, 200)
    ps = probability_zero(omega, tau_fine, t2, shape, visibility)
    env = 0.5 * visibility * np.exp(-(tau_fine / t2) ** shape)
    
    fig_fringe = go.Figure()
    fig_fringe.add_trace(go.Scatter(x=tau_fine, y=ps, mode='lines', name='P(0)'))
    fig_fringe.add_trace(go.Scatter(x=tau_fine, y=0.5+env, mode='lines', line=dict(dash='dot', color='gray'), name='Envelope'))
    fig_fringe.add_trace(go.Scatter(x=tau_fine, y=0.5-env, mode='lines', line=dict(dash='dot', color='gray'), showlegend=False))
    
    if st.button("Run paired experiment", type="primary"):
        counts, n, rep_counts = simulate(omega, noise, repeats, np.random.default_rng(seed), return_repeat_counts=True)
        f_obs = counts / n
        fig_fringe.add_trace(go.Scatter(x=TAUS_US, y=f_obs, mode='markers', marker=dict(size=8, color='red'), name=f'Observed (k={repeats})'))
        
        st.plotly_chart(fig_fringe, use_container_width=True)
        
        st.subheader("2. Posteriors & Estimates")
        
        results = {}
        results["B1 (Assumed)"] = bayes_summary(counts, n, "assumed")
        results["B2 (T2*/V margin.)"] = bayes_summary(counts, n, "nuisance")
        results["B3 (+Shape margin.)"] = bayes_summary(counts, n, "nuisance_shape")
        
        if NN_AVAILABLE and Path("artifacts/nn_weights.npz").exists():
            load_weights("artifacts/nn_weights.npz")
            results["NN (Amortized)"] = nn_summary(counts, n, repeat_counts=rep_counts)
            
        fig_post = go.Figure()
        cols = st.columns(len(results))
        
        for (name, res), col in zip(results.items(), cols):
            b_grid = b_from_omega(res["grid"])
            post = res["posterior"]
            fig_post.add_trace(go.Scatter(x=b_grid, y=post/post.max(), mode='lines', name=name))
            
            with col:
                est_b = b_from_omega(res["omega_rad_per_us"])
                error = est_b - field
                st.metric(name, f"{est_b:.3f} µT", f"{error:+.3f} error", delta_color="inverse")
                st.write(f"**68% CI:** [{b_from_omega(res['ci68'][0]):.3f}, {b_from_omega(res['ci68'][1]):.3f}]")
                st.write(f"**95% CI:** [{b_from_omega(res['ci95'][0]):.3f}, {b_from_omega(res['ci95'][1]):.3f}]")
                
        fig_post.add_vline(x=field, line_dash="dash", line_color="black", annotation_text="True B")
        fig_post.update_layout(title="Posterior distributions", xaxis_title="Magnetic field (µT)", yaxis_title="Normalized Posterior")
        st.plotly_chart(fig_post, use_container_width=True)
    else:
        st.plotly_chart(fig_fringe, use_container_width=True)

with tab_bench:
    st.subheader("Precomputed Benchmark Results")
    bench_file = Path("results/benchmark.json")
    if bench_file.exists():
        data = json.loads(bench_file.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
        
        st.markdown("**Error vs Sensing Time (Matched Noise)**")
        df_match = df[df["scenario"] == "matched"].copy()
        if not df_match.empty:
            df_match["Method"] = df_match["method"].replace({
                "bayes_assumed": "B1", "bayes_nuisance": "B2", 
                "bayes_nuisance_shape": "B3", "nn": "NN"
            })
            fig_err = px.line(df_match, x="t_total_ms", y="rmse_uT", color="Method", markers=True, log_y=True)
            st.plotly_chart(fig_err, use_container_width=True)
            
        st.markdown("**Coverage vs Phase Drift (k=16)**")
        df_drift = df[(df["axis"] == "drift") & (df["k"] == 16)].copy()
        if not df_drift.empty:
            df_drift["Method"] = df_drift["method"].replace({
                "bayes_assumed": "B1", "bayes_nuisance": "B2", 
                "bayes_nuisance_shape": "B3", "nn": "NN"
            })
            fig_cov = px.line(df_drift, x="value", y="cov95", color="Method", markers=True)
            fig_cov.add_hline(y=0.95, line_dash="dash", annotation_text="Nominal (95%)")
            fig_cov.add_hline(y=0.90, line_dash="dot", line_color="gray", annotation_text="Threshold")
            st.plotly_chart(fig_cov, use_container_width=True)
        st.info("Run `python -m bench.run_benchmark` to populate this page.")

st.info("Note on custom data: The NN requires per-repeat counts to extract over-dispersion features. If you upload a standard pooled-count CSV (tau, n_shots, n_zero), the NN inference will be automatically disabled and only B1/B2/B3 will run.")

