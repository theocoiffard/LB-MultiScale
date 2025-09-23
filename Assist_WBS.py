import streamlit as st
import numpy as np
import pandas as pd
import json


# To run via terminal : streamlit run Assist_WBS.py


st.set_page_config(layout="wide")

# === Helpers format/round ===
def fmt6(x):
    """Retourne x en notation scientifique avec 6 décimales (chaîne)."""
    try:
        return f"{float(x):.6e}"
    except Exception:
        return str(x)

def fmt6_if_num(x):
    """Applique fmt6 si numérique, sinon renvoie tel quel (utile pour '-')"""
    try:
        return f"{float(x):.6e}"
    except Exception:
        return x

# === Fonctions physiques/LB ===
def eps3(mu, rho, dt, dx):
    cs = dx / dt / np.sqrt(3)
    return 1 / (mu / (rho * cs**2 * dt) + 1/2)

def eps2(mu, rho, dt, phi, k, kn, dx):
    cs = dx / dt / np.sqrt(3)
    sv_adi = 1 / (mu / (phi * rho * cs**2 * dt) + 1/2)
    theta_adi = dt * mu / (2 * k * rho)
    return sv_adi, theta_adi, np.log(theta_adi) / np.log(kn)

def eps1(mu, rho, dt, phi, k, kn, dx):
    cs = dx / dt / np.sqrt(3)
    theta_adi = (dt * mu / k) / (dt * mu / k + 2 * rho)
    sv_adi = 1 / (((1 - theta_adi) * mu / phi) / (rho * cs**2 * dt) + 1/2)
    return sv_adi, theta_adi, np.log(theta_adi) / np.log(kn)

# === Layout ===
col1, col2 = st.columns([1.2, 2])  # Gauche: paramètres, Droite: résultats

with col1:
    st.header("🔧 Physicals Parameters")

    triplet = st.selectbox("Triplet :", ['rho_u_L', 'rho_u_k', 'rho_mu_L', 'rho_mu_k', 'u_mu_L'])

    def param_line(name, val, order_min, order_max, order_init):
        col_a, col_b = st.columns([1, 2])
        with col_a:
            factor = st.number_input(f"{name} (mantissa)", value=val, step=0.1)
        with col_b:
            ordre = st.slider(f"{name} (order) ", order_min, order_max, order_init, key=name)
        return factor, ordre

    rho_factor, order_rho = param_line("ρ", 1.2, -3, 3, 0)
    u_factor, order_u = param_line("u", 2.0, -7, 4, 0)
    k_factor, order_k = param_line("k", 1.0, -14, 6, -7)
    mu_factor, order_mu = param_line("μ", 2.2, -6, 0, -5)
    L_factor, order_L = param_line("L", 5.0, -7, 4, -3)

    phi = st.slider("Φ", 0.01, 1.0, 0.9)
    Nx = st.slider("Nx", 30, 1000, 60)
    same_dt_dx = st.checkbox("Δt = Δx", value=True)
    c_slider = int(st.slider("c = Δx / Δt", 1, 10, 1, step=1))

with col2:
    st.header("📊 Results")

    # === Physique ===
    rho_physic = rho_factor * 10**order_rho
    u_physic = u_factor * 10**order_u
    k_physic = k_factor * 10**order_k
    mu_physic = mu_factor * 10**order_mu
    L_physic = L_factor * 10**order_L

    if triplet == 'rho_u_L':
        rho_ref = rho_physic
        u_ref = u_physic
        k_ref = L_physic**2
        mu_ref = rho_physic * u_physic * L_physic
        L_ref = L_physic
    elif triplet == 'rho_u_k':
        rho_ref = rho_physic
        u_ref = u_physic
        k_ref = k_physic
        mu_ref = rho_physic * u_physic * np.sqrt(k_physic)
        L_ref = np.sqrt(k_physic)
    elif triplet == 'rho_mu_L':
        rho_ref = rho_physic
        mu_ref = mu_physic
        L_ref = L_physic
        u_ref = mu_physic / (rho_physic * L_physic)
        k_ref = L_physic**2
    elif triplet == 'rho_mu_k':
        rho_ref = rho_physic
        mu_ref = mu_physic
        k_ref = k_physic
        L_ref = np.sqrt(k_physic)
        u_ref = mu_physic / (rho_physic * np.sqrt(k_physic))
    elif triplet == 'u_mu_L':
        u_ref = u_physic
        mu_ref = mu_physic
        L_ref = L_physic
        rho_ref = mu_physic / (u_physic * L_physic)
        k_ref = L_physic**2

    dx_ref = L_ref / Nx
    dt_ref = dx_ref if same_dt_dx else dx_ref / c_slider

    # === Adimensionné ===
    rho_adi = rho_physic / rho_ref
    u_adi = u_physic / u_ref
    k_adi = k_physic / k_ref
    mu_adi = mu_physic / mu_ref
    L_adi = L_physic / L_ref
    dx_adi = L_adi / Nx
    dt_adi = dx_adi if same_dt_dx else dx_adi / c_slider
    kn_adi = dx_adi / L_adi
    cs_adi = dx_adi / dt_adi / np.sqrt(3)

    # === LB orders ===
    sv_eps3 = eps3(mu_adi, rho_adi, dt_adi, dx_adi)
    sv_eps2, theta_eps2, log_theta_eps2 = eps2(mu_adi, rho_adi, dt_adi, phi, k_adi, kn_adi, dx_adi)
    sv_eps1, theta_eps1, log_theta_eps1 = eps1(mu_adi, rho_adi, dt_adi, phi, k_adi, kn_adi, dx_adi)

    # === Poiseuille ===
    delta_p = 8 * mu_physic * u_physic / L_physic
    p_out = rho_physic * cs_adi**2
    p_in = p_out + delta_p
    rho_in = p_in / cs_adi**2

    delta_p_ref = 8 * mu_ref * u_ref / L_ref
    p_out_ref = rho_ref * cs_adi**2
    p_in_ref = p_out_ref + delta_p_ref
    rho_in_ref = p_in_ref / cs_adi**2

    delta_p_adi = 8 * mu_adi * u_adi / L_adi
    p_out_adi = rho_adi * cs_adi**2
    p_in_adi = p_out_adi + delta_p_adi
    rho_in_adi = p_in_adi / cs_adi**2

    # === Darcy velocity ===
    u_Darcy = (k_physic / mu_physic) * (delta_p / L_physic)
    u_Darcy_adi = u_Darcy / u_ref

    # === Tableau fusion Phys/Ref/Adim ===
    data = {
        "Variables": [
            "ρ", "u", "k", "μ", "L", "Δx", "Δt",
            "Δp", "ρ_out", "p_out", "p_in", "ρ_in", "u_Darcy", "Φ", "Nx", "c"
        ],
        "Physical": [
            rho_physic, u_physic, k_physic, mu_physic, L_physic, dx_ref, dt_ref,
            delta_p, rho_physic, p_out, p_in, rho_in, u_Darcy, phi, Nx, c_slider
        ],
        "Reference": [
            rho_ref, u_ref, k_ref, mu_ref, L_ref, dx_ref, dt_ref,
            delta_p_ref, rho_ref, p_out_ref, p_in_ref, rho_in_ref, u_Darcy / u_Darcy, phi, Nx, c_slider
        ],
        "No dimension": [
            rho_adi, u_adi, k_adi, mu_adi, L_adi, dx_adi, dt_adi,
            delta_p_adi, rho_adi, p_out_adi, p_in_adi, rho_in_adi, u_Darcy_adi, phi, Nx, c_slider
        ]
    }

    df_fusion = pd.DataFrame(data)

    # === Nombres adimensionnels ===
    Re = rho_adi * u_adi * L_adi / mu_adi
    Re_k = rho_adi * u_adi * np.sqrt(k_adi) / mu_adi
    Da = k_adi / L_adi**2
    Ma = u_adi / cs_adi
    Kn = dx_adi / L_adi  # = epsilon

    # === Adimensional numbers with u_Darcy ===
    Re_Darcy = rho_adi * u_Darcy_adi * L_adi / mu_adi
    Re_k_Darcy = rho_adi * u_Darcy_adi * np.sqrt(k_adi) / mu_adi
    Ma_Darcy = u_Darcy_adi / cs_adi

    df_adim = pd.DataFrame({
        "Number": ["Re", "Re_k", "Kn", "Da", "Ma"],
        "With u": [Re, Re_k, Kn, Da, Ma],
        "With u_Darcy": [Re_Darcy, Re_k_Darcy, "-", "-", Ma_Darcy]
    })

    # === Règle de décision pour s_v / s_j
    eps1_alpha = float(log_theta_eps1)
    eps2_alpha = float(log_theta_eps2)
    epsilon = kn_adi  # = dx / L

    if eps2_alpha >= 3:
        sv_true = sv_eps3
        alpha_true = eps2_alpha
        sj_true = '-'
    elif eps1_alpha >= 2:
        sv_true = sv_eps2
        alpha_true = eps2_alpha
        sj_true = "-"
    elif eps2_alpha < 2:
        try:
            beta = np.log(1 - epsilon ** eps1_alpha) / np.log(epsilon)
        except Exception:
            beta = np.nan

        if beta >= 1:
            sv_true = sv_eps1
            alpha_true = eps1_alpha
            sj_true = -dt_adi / rho_adi * k_adi / mu_adi + 2
        else:
            sv_true = sv_eps1
            alpha_true = eps1_alpha
            sj_true = "-"
    else:
        sv_true = "-"
        alpha_true = "-"
        sj_true = "-"

    df_eps = pd.DataFrame({
        "LB-WBS model": ["O(eps3)", "O(eps2)", "O(eps1)"],
        "s_v": [sv_eps3, sv_eps2, sv_eps1],
        "alpha": ["-", log_theta_eps2, log_theta_eps1],
        "s_j": ["-", "-", sj_true]
    })

    # ➕ Ligne sélectionnée
    df_eps.loc[len(df_eps)] = {
        "LB-WBS model": "Selected",
        "s_v": sv_true if sv_true is not None else "-",
        "alpha": alpha_true if alpha_true is not None else "-",
        "s_j": sj_true if sj_true is not None else "-"
    }

    # === FORMATAGE EN NOTATION SCIENTIFIQUE (AFFICHAGE)
    df_fusion_fmt = df_fusion.copy()
    df_fusion_fmt[["Physical","Reference","No dimension"]] = df_fusion_fmt[["Physical","Reference","No dimension"]].applymap(fmt6)

    df_adim_fmt = df_adim.copy()
    df_adim_fmt["With u"] = df_adim_fmt["With u"].map(fmt6_if_num)
    df_adim_fmt["With u_Darcy"] = df_adim_fmt["With u_Darcy"].map(fmt6_if_num)

    df_eps_fmt = df_eps.copy()
    for col in ["s_v", "alpha", "s_j"]:
        df_eps_fmt[col] = df_eps_fmt[col].map(fmt6_if_num)

    # === Affichages
    st.subheader("🧮 Parameters set")
    st.dataframe(df_fusion_fmt, use_container_width=True)

    col_adim, col_eps = st.columns(2)
    with col_adim:
        st.subheader("📐 Adimensionnal numbers")
        st.table(df_adim_fmt)

    with col_eps:
        st.subheader("⚙️ LB-WBS model parameters")
        st.table(df_eps_fmt)

    # === Export ===
    st.markdown("### 💾 Export Results")
    filename = st.text_input("File name (without extension)", value="simulation_input")

    # Mapping des noms de variables
    variable_mapping = {
        "ρ": "rho",
        "u": "u",
        "k": "k",
        "μ": "mu",
        "L": "L",
        "Δx": "dx",
        "Δt": "dt",
        "Δp": "delta_p",
        "ρ_out": "rho_out",
        "p_out": "p_out",
        "p_in": "p_in",
        "ρ_in": "rho_in",
        "u_Darcy": "u_Darcy",
        "Φ": "phi",
        "Nx": "Nx",
        "c": "c"
    }

    df_fusion_fmt_export = df_fusion_fmt.copy()
    df_fusion_fmt_export["Variables"] = df_fusion_fmt_export["Variables"].map(variable_mapping)

    # Idem formatage pour export JSON
    df_adim_fmt_export = df_adim_fmt.copy()
    df_eps_fmt_export = df_eps_fmt.copy()

    results = {
        "physicals": df_fusion_fmt_export.to_dict(orient="list"),
        "adimensional_numbers": df_adim_fmt_export.to_dict(orient="list"),
        "LB_WBS_parameters": df_eps_fmt_export.to_dict(orient="list")
    }

    json_data = json.dumps(results, indent=2)

    if filename.strip():
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"{filename.strip()}.json",
            mime="application/json"
        )
    else:
        st.info("📝 Please enter a file name to enable downloading.")
