"""
sailswarm_toy_forces.py
-----------------------
Minimal, code-ready equations for a robotic sailboat simulator.

Includes:
- Apparent wind calculation
- Simple sail polar model (CL, CD) and force decomposition (drive/side)
- Calm-water quadratic drag (placeholder)
- Added resistance in waves (very coarse, tunable placeholder)
- Morison-type force for slender elements (e.g., mast/struts) in waves

All angles in radians unless stated otherwise.
PutTogether by: Some Papers, Pranav and GPT
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Tuple

AIR_DENSITY = 1.225         # kg/m^3 (sea level, 15 C)
WATER_DENSITY = 1025.0      # kg/m^3 (seawater)
GRAVITY = 9.80665

# ----------------------
# Kinematics & helpers
# ----------------------

def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    x = (angle + math.pi) % (2.0*math.pi)
    return x - math.pi

def apparent_wind(V_tw: float, beta_tw: float, V_boat: float, psi_boat: float) -> Tuple[float, float]:
    """
    Compute apparent wind speed and direction in the *boat-fixed* frame.

    Args:
        V_tw: true wind speed [m/s] (over ground)
        beta_tw: true wind direction [rad] in earth frame, 0 = +x (east), CCW positive
        V_boat: boat speed over ground [m/s]
        psi_boat: boat heading [rad] in earth frame, 0 = +x (east), CCW positive

    Returns:
        (V_app, beta_app): apparent wind speed [m/s] and angle [rad] relative to boat x-axis (bow).
                           beta_app = 0 -> headwind from straight ahead; +CCW (port).
    """
    # Earth-frame wind vector
    Wx = V_tw * math.cos(beta_tw)
    Wy = V_tw * math.sin(beta_tw)
    # Earth-frame boat velocity
    Bx = V_boat * math.cos(psi_boat)
    By = V_boat * math.sin(psi_boat)
    # Apparent wind in earth frame: W - B
    Ax = Wx - Bx
    Ay = Wy - By
    # Rotate into boat frame (x-forward, y-port)
    c, s = math.cos(psi_boat), math.sin(psi_boat)
    Axb =  c*Ax + s*Ay
    Ayb = -s*Ax + c*Ay
    V_app = math.hypot(Axb, Ayb)
    beta_app = math.atan2(Ayb, Axb)  # angle of incoming flow relative to boat x-axis
    return V_app, wrap_pi(beta_app)

# ----------------------
# Sail aerodynamics
# ----------------------

@dataclass
class SailPolarParams:
    CL_alpha: float = 2*math.pi * 0.6   # effective lift slope [1/rad], reduced from 2π for finite aspect & viscous effects
    alpha_stall: float = math.radians(18.0)  # stall angle (|alpha|) for soft sails (rough ballpark)
    CD0: float = 0.02                   # parasitic drag at zero lift
    k_induced: float = 0.06             # induced drag factor (Parabolic: CD = CD0 + k*CL^2)

def sail_polar(alpha: float, p: SailPolarParams = SailPolarParams()) -> Tuple[float, float]:
    """
    Very simple polar: linear CL up to stall; CD parabolic in CL.
    Args:
        alpha: angle of attack [rad]
        p: sail polar parameters
    Returns:
        (CL, CD)
    """
    # Symmetric clipping around 0 for a crude stall model
    alpha_eff = max(-p.alpha_stall, min(p.alpha_stall, alpha))
    CL = p.CL_alpha * alpha_eff
    CD = p.CD0 + p.k_induced * CL*CL
    return CL, CD

def sail_forces(
    A_sail: float,
    rho_air: float,
    V_app: float,
    alpha: float,
    beta_app: float,
    CE_height: float = 1.0,
    polar: SailPolarParams = SailPolarParams(),
) -> Tuple[Tuple[float, float], float]:
    """
    Compute sail Lift/Drag and resolve into boat axes: Drive (+x) and Side (+y port).
    Also returns a simple heeling moment about the roll axis from side force.
    Args:
        A_sail: trimmed sail area [m^2]
        rho_air: air density [kg/m^3]
        V_app: apparent wind speed [m/s]
        alpha: AoA [rad] (sheeting/trim + geometry – you choose how to compute it from beta_app)
        beta_app: apparent wind angle in boat frame [rad]
        CE_height: center-of-effort height above waterline [m]
        polar: SailPolarParams
    Returns:
        ((F_drive, F_side), M_heel)
            F_drive: along +x (forward), F_side: +y (to port). Signs follow standard boat axes.
            M_heel: rolling moment [N·m], positive heeling to starboard under port-side force.
    """
    CL, CD = sail_polar(alpha, polar)
    q = 0.5 * rho_air * V_app*V_app
    L = q * A_sail * CL
    D = q * A_sail * CD
    # Lift is ~perpendicular to relative flow, Drag along flow.
    # Resolve L,D into boat axes via beta_app
    # Unit vectors: e_flow = [cos(beta_app), sin(beta_app)]
    cb, sb = math.cos(beta_app), math.sin(beta_app)
    # Drag along flow (opposes the flow direction acting on the sail)
    Fx_D = -D * cb
    Fy_D = -D * sb
    # Lift perpendicular to flow (90 deg CCW from flow dir)
    Fx_L = -L * (-sb)  # = L*sb
    Fy_L = -L * (cb)   # = -L*cb
    Fx = Fx_D + Fx_L   # drive (+ forward)
    Fy = Fy_D + Fy_L   # side (+ port)
    M_heel = Fy * CE_height
    return (Fx, Fy), M_heel

# ----------------------
# Hull hydrodynamics (very simple placeholders)
# ----------------------

def calm_water_quadratic_drag(rho_water: float, C_D_hull: float, A_ref: float, U: float) -> float:
    """
    Quadratic drag placeholder: R = 0.5 * rho * C_D * A_ref * U^2
    Use a reference area you are comfortable with (e.g., wetted area or frontal proj.).
    """
    return 0.5 * rho_water * C_D_hull * A_ref * U*U

def added_resistance_waves_simple(
    U: float,
    rho_water: float,
    A_waterplane: float,
    wave_amp: float,
    wave_length: float,
    wave_heading_rel: float,
    k_raw: float = 1.0
) -> float:
    """
    Extremely simple, tunable placeholder for Added Resistance in Waves (RAW).
    Scales with wave steepness^2 and vanishes with following seas factor.
    Args:
        U: boat speed [m/s]
        rho_water: [kg/m^3]
        A_waterplane: waterplane area [m^2] (proxy for wave interaction strength)
        wave_amp: wave amplitude a [m] (Hs ≈ 4a for narrowband)
        wave_length: wave length λ [m]
        wave_heading_rel: wave direction relative to bow [rad] (0=head seas, pi=following)
        k_raw: tuning coefficient (order 0.2–2); fit to data or higher-fidelity model.
    Returns:
        Added resistance [N], non-negative.
    Notes:
        RAW_true depends on 2nd-order wave-drift forces and motions. This is a coarse proxy:
            RAW ~ (0.5*rho*U^2)*A_wp*(ka)^2 * f(heading)
        with steepness k*a = (2π/λ)*a and f(0)=1, f(π)=0.
    """
    if wave_length <= 0.0:
        return 0.0
    k = 2.0*math.pi / wave_length
    steep2 = (k * wave_amp)**2
    # Simple heading factor (0..1), max in head seas, ~0 in following
    f_head = 0.5 * (1.0 + math.cos(wave_heading_rel))  # 1 at 0, 0 at pi
    base = 0.5 * rho_water * U*U * A_waterplane
    RAW = k_raw * base * steep2 * f_head
    return max(0.0, RAW)

# ----------------------
# Morison-type force (slender elements)
# ----------------------

def morison_inline_force(
    rho: float,
    C_m: float,
    C_d: float,
    D_char: float,
    L_char: float,
    u: float,
    u_dot: float
) -> float:
    """
    Inline force on a slender cylinder segment in oscillatory flow (Morison eq.).
    F = rho*C_m*V*u_dot + 0.5*rho*C_d*A*u*|u|
    Args:
        rho: fluid density [kg/m^3]
        C_m: inertia coefficient (~2 for cylinders; tune/fit)
        C_d: drag coefficient (order 0.6–1.2; tune/fit)
        D_char: characteristic diameter [m]
        L_char: span length [m]
        u: inline fluid-particle velocity relative to body [m/s]
        u_dot: inline acceleration [m/s^2]
    Returns:
        Force [N] in the inline direction (positive with +u).
    """
    V = math.pi * (D_char**2) / 4.0 * L_char
    A = D_char * L_char
    F_inertia = rho * C_m * V * u_dot
    F_drag = 0.5 * rho * C_d * A * u * abs(u)
    return F_inertia + F_drag

# ----------------------
# Convenience: combined resistances
# ----------------------

def total_longitudinal_force(
    Fx_drive: float,
    U: float,
    rho_water: float,
    C_D_hull: float,
    A_ref: float,
    RAW: float = 0.0
) -> float:
    """
    Net surge force (positive forward): drive - calm-water drag - added resistance in waves.
    """
    return Fx_drive - calm_water_quadratic_drag(rho_water, C_D_hull, A_ref, U) - RAW

# ----------------------
# Example usage 
# ----------------------
if __name__ == "__main__":
    # Example numbers (replace with your boat):
    V_tw = 6.0              # m/s true wind
    beta_tw = math.radians(60.0)  # from starboard bow
    V_boat = 2.0            # m/s
    psi_boat = math.radians(0.0)
    V_app, beta_app = apparent_wind(V_tw, beta_tw, V_boat, psi_boat)

    A_sail = 1.2            # m^2 (robotic boat)
    alpha = math.radians(8.0)
    CE_h = 0.7
    (Fx, Fy), Mheel = sail_forces(A_sail, AIR_DENSITY, V_app, alpha, beta_app, CE_h)

    # Hydrodynamics
    U = V_boat
    C_D_hull = 0.9
    A_ref = 0.05            # m^2, proxy frontal area
    # Simple waves
    a = 0.15                # wave amplitude [m]
    lam = 8.0               # wavelength [m]
    beta_wave_rel = 0.0     # head seas
    RAW = added_resistance_waves_simple(U, WATER_DENSITY, A_waterplane=0.25, wave_amp=a, wave_length=lam, wave_heading_rel=beta_wave_rel, k_raw=0.6)

    F_net = total_longitudinal_force(Fx, U, WATER_DENSITY, C_D_hull, A_ref, RAW)

    print("Apparent wind: V=%.2f m/s, beta=%.1f deg" % (V_app, math.degrees(beta_app)))
    print("Sail forces: Drive=%.1f N, Side=%.1f N, Heel=%.1f N·m" % (Fx, Fy, Mheel))
    print("Added resistance in waves (toy): %.2f N" % RAW)
    print("Net surge force: %.2f N" % F_net)
