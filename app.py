
import math
from typing import Dict, Tuple, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Constants
# -----------------------------
R_L_BAR = 0.08314462618  # L·bar/(mol·K)
M_CO2 = 44.01            # g/mol
P_ATM_BAR = 1.01325
T_REF_K = 273.15         # "volumes" reference (1 bar, 0°C)

# -----------------------------
# Henry's law utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def henry_lookup_table() -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate Henry's-law solubility coefficient for CO₂ in pure water:
        C* = kH(T) * P_CO2
    Units: kH in mol/(L·bar)

    These are representative values for estimation. Real beverages (sugars/salts)
    generally reduce CO₂ solubility ("salting-out").
    """
    T_C = np.array([0, 5, 10, 15, 20, 25], dtype=float)
    kH = np.array([0.077, 0.064, 0.054, 0.046, 0.040, 0.034], dtype=float)
    return T_C, kH


def kH_CO2_water_mol_per_L_bar(T_c: float) -> float:
    """Linear interpolation of kH from the built-in lookup table (clamped to table range)."""
    T_grid, kH_grid = henry_lookup_table()
    T_c_clamped = float(np.clip(T_c, T_grid.min(), T_grid.max()))
    return float(np.interp(T_c_clamped, T_grid, kH_grid))


# -----------------------------
# Unit conversions
# -----------------------------
def pressure_abs_bar(p_value: float, mode: str) -> float:
    """Convert user pressure to absolute bar."""
    return p_value + P_ATM_BAR if mode == "Gauge (bar g)" else p_value


def target_to_mol_per_L(value: float, unit: str) -> float:
    """Convert target carbonation to mol/L."""
    if unit == "g/L CO₂":
        return value / M_CO2
    if unit == "mol/L":
        return value
    # Volumes: liters of CO₂ at 1 bar and 0°C per liter of beverage
    return value * (1.0 / (R_L_BAR * T_REF_K))


def mol_per_L_to_unit(c_mol_per_L: float, unit: str) -> float:
    """Convert mol/L to the selected display unit."""
    if unit == "g/L CO₂":
        return c_mol_per_L * M_CO2
    if unit == "mol/L":
        return c_mol_per_L
    return c_mol_per_L * (R_L_BAR * T_REF_K)


# -----------------------------
# Model
# -----------------------------
def time_to_target_seconds(c_star: float, c_target: float, kla: float) -> Optional[float]:
    """
    C(t)=C*(1-exp(-kLa t)) => t = -(1/kLa)*ln(1 - Ctarget/C*)
    Returns None if not achievable or invalid kla.
    """
    if kla <= 0:
        return None
    if c_target <= 0:
        return 0.0
    if c_target >= c_star:
        return None
    return -math.log(1.0 - (c_target / c_star)) / kla


def make_curve(c_star: float, kla: float, t_max: float, n: int = 400):
    t = np.linspace(0, max(t_max, 1e-6), n)
    c = c_star * (1.0 - np.exp(-kla * t))
    return t, c


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Carbonation Time Estimator", layout="wide")
st.title("ZEF - Carbonation Time Estimator (Confidential)")
st.caption("Henry’s Law + kLa kinetics to estimate time-to-carbonation (order-of-magnitude engineering estimator).")

with st.sidebar:
    st.header("Inputs")

    st.subheader("Pressure")
    pressure_mode = st.radio("Pressure type", ["Gauge (bar g)", "Absolute (bar abs)"], index=0)
    p_in = st.number_input("Carbonation pressure", min_value=0.0, max_value=60.0, value=4.0, step=0.1)
    p_abs = pressure_abs_bar(p_in, pressure_mode)

    st.subheader("Temperature")
    T_c = st.number_input("Liquid temperature (°C)", min_value=-1.0, max_value=40.0, value=10.0, step=0.5)

    st.subheader("Target carbonation")
    target_unit = st.selectbox("Target unit", ["g/L CO₂", "Volumes of CO₂", "mol/L"], index=0)
    default_target = 8.0 if target_unit == "g/L CO₂" else (4.0 if target_unit == "Volumes of CO₂" else 8.0 / M_CO2)
    c_target_in = st.number_input("Desired carbonation level", min_value=0.0, max_value=100.0, value=float(default_target), step=0.1)

    st.subheader("Mass transfer (kLa)")
    preset = st.selectbox(
        "kLa preset (very rough estimates for typical home carbonation methods)",
        [
            "Static + diffuser stone",
            "Gentle rocking/shaking",
            "Recirculation loop + stone",
            "SodaStream-like violent injection (effective)",
        ],
        index=0,
    )

    presets: Dict[str, Tuple[float, float, float]] = {
        "Static + diffuser stone": (0.005, 0.02, 0.01),
        "Gentle rocking/shaking": (0.02, 0.08, 0.04),
        "Recirculation loop + stone": (0.05, 0.2, 0.1),
        "SodaStream-like violent injection (effective)": (0.5, 5.0, 1.0),
    }
    kla_min, kla_max, kla_def = presets[preset]

    use_custom = st.checkbox("Custom kLa (choose your own to match data)", value=False)
    if use_custom:
        kla = st.number_input("kLa (s⁻¹)", min_value=1e-6, max_value=10.0, value=float(kla_def), step=0.001, format="%.4f")
    else:
        step = float((kla_max - kla_min) / 200) if kla_max > kla_min else 0.001
        kla = st.slider("kLa (s⁻¹)", min_value=float(kla_min), max_value=float(kla_max), value=float(kla_def), step=step)

    st.subheader("Optional corrections")
    bev_factor = st.slider(
        "Non-ideal beverage factor (solubility multiplier)",
        min_value=0.80,
        max_value=1.00,
        value=1.00,
        step=0.01,
        help="Sugars/salts reduce CO₂ solubility. 1.00 ≈ pure water; lower values mimic 'salting-out'.",
    )

# -----------------------------
# Calculations
# -----------------------------
kH = kH_CO2_water_mol_per_L_bar(T_c) * bev_factor      # mol/(L·bar)
p_co2 = p_abs                                          # assume 100% CO2 headspace
c_star = kH * p_co2                                    # mol/L
c_target = target_to_mol_per_L(c_target_in, target_unit)

t_req = time_to_target_seconds(c_star, c_target, kla)

t90 = 2.303 / kla if kla > 0 else float("nan")
t95 = 2.996 / kla if kla > 0 else float("nan")

c_star_disp = mol_per_L_to_unit(c_star, target_unit)
c_target_disp = mol_per_L_to_unit(c_target, target_unit)

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Equilibrium $C^*$", f"{c_star_disp:,.2f} {target_unit}")
with col2:
    st.metric("Target $C_{target}$", f"{c_target_disp:,.2f} {target_unit}")
with col3:
    st.metric("Achievable?", "Yes ✅" if (kla > 0 and c_target < c_star) else "No ❌")
with col4:
    if t_req is not None and kla > 0 and c_target < c_star:
        st.metric("Estimated time", f"{t_req:,.0f} s", f"{t_req/60:,.2f} min")
    else:
        st.metric("Estimated time", "—", "Not achievable / invalid")

# Guidance if not achievable
if kla <= 0:
    st.error("kLa must be > 0.")
elif c_target >= c_star and c_target > 0:
    p_min_abs = c_target / kH if kH > 0 else float("inf")
    p_min_g = max(p_min_abs - P_ATM_BAR, 0.0)
    st.warning(
        f"Target is **not achievable** at the selected temperature and pressure.\n\n"
        f"Minimum required pressure (assuming pure CO₂ headspace): **{p_min_abs:.2f} bar abs** "
        f"(≈ **{p_min_g:.2f} bar g**)."
    )

st.info(
    "Safety: Use pressure-rated vessels/fittings and a PRV. Real beverages differ from pure water. "
    "Opening/transfer can lose CO₂; counter-pressure filling helps retain carbonation."
)

# -----------------------------
# Plot
# -----------------------------
st.subheader("Carbonation curve")

if kla > 0:
    if t_req is not None and c_target < c_star:
        horizon = max(2.0 * t_req, 60.0)
    else:
        horizon = 600.0

    t, c = make_curve(c_star, kla, horizon)
    c_disp = np.array([mol_per_L_to_unit(x, target_unit) for x in c])

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(t, c_disp)
    ax.axhline(c_target_disp, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Dissolved CO₂ ({target_unit})")
    ax.set_title("C(t) = C* (1 - exp(-kLa·t))")

    if t_req is not None and c_target < c_star:
        ax.axvline(t_req, linestyle=":")
        ax.text(t_req, c_target_disp, f"  t ≈ {t_req:.0f} s", va="bottom")

    st.pyplot(fig, clear_figure=True)

with st.expander("Diagnostics: time constants & assumptions", expanded=False):
    st.markdown(
        f"""
**Model**
- $C(t)=C^*(1-e^{{-k_La t}})$ with $C^* = k_H(T)P_{{CO_2}}$

**Assumptions**
- Headspace is **100% CO₂**, so $P_{{CO_2}} \\approx P_{{abs}}$.
- Henry coefficient uses a built-in lookup table for **pure water** (0–25 °C) with linear interpolation.
- “Non-ideal beverage factor” crudely accounts for reduced solubility in sugary/salty drinks.

**Inputs & Physical Meaning**

- $P_{{abs}}$ = **{p_abs:.3f} bar (absolute pressure)**  
  Absolute CO₂ pressure in the headspace.  
  Units: **bar (abs)**.  
  Assumed $P_{{CO_2}} \\approx P_{{abs}}$ (100% CO₂ headspace).

- $T$ = **{T_c:.1f} °C**  
  Liquid temperature.  
  Units: **°C** (internally converted to K where needed).

- $k_H(T)$ = **{kH:.5f} mol·L⁻¹·bar⁻¹**  
  Henry’s-law solubility coefficient for CO₂ in the liquid.  
  Governs equilibrium solubility:  
  $C^* = k_H(T)P_{{CO_2}}$

- $k_La$ = **{kla:.4f} s⁻¹**  
  Volumetric mass transfer coefficient controlling approach to equilibrium.

---

### Separation of $k_L$ and $a$

\[
k_La = k_L \cdot a
\]

- $k_L$ (m·s⁻¹) → Liquid-side mass transfer coefficient  
  Depends on:
  - CO₂ diffusivity  
  - Liquid viscosity  
  - Turbulence / mixing  
  - Temperature  

- $a$ (m²·m⁻³ = m⁻¹) → Gas–liquid interfacial area per volume  
  Depends on:
  - Bubble size (smaller bubbles → larger $a$)  
  - Gas holdup  
  - Diffuser pore size  
  - Agitation / recirculation  

---

**Interpretation**

- $k_H(T)$ sets the equilibrium ceiling.  
- $k_La$ sets the speed to reach that ceiling.


**Characteristic times** (depend only on $k_La$)

- $t_{{25}} = 0.288/k_La$ = **{0.288/kla:.1f} s** (≈ **{(0.288/kla)/60:.2f} min**)
- $t_{{50}} = 0.693/k_La$ = **{0.693/kla:.1f} s** (≈ **{(0.693/kla)/60:.2f} min**)
- $t_{{75}} = 1.386/k_La$ = **{1.386/kla:.1f} s** (≈ **{(1.386/kla)/60:.2f} min**)
- $t_{{90}} = 2.303/k_La$ = **{t90:.1f} s** (≈ **{t90/60:.2f} min**)
- $t_{{95}} = 2.996/k_La$ = **{t95:.1f} s** (≈ **{t95/60:.2f} min**)

"""
    )

with st.expander("Henry’s-law lookup table used (pure water)", expanded=False):
    T_grid, kH_grid = henry_lookup_table()
    st.dataframe({"T (°C)": T_grid, "kH (mol/(L·bar))": kH_grid}, use_container_width=True)
