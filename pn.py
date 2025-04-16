import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st

# Constants
q = 1.6e-19      # Charge of electron (C)
eps_si = 11.7 * 8.85e-12  # Permittivity of silicon (F/m)
k = 1.38e-23     # Boltzmann constant (J/K)
T = 300          # Temperature (K)
Vt = k * T / q   # Thermal voltage

# Function to compute PN junction properties
def simulate_pn_junction(Vbias):
    # Doping levels (change these for fun!)
    NA = 1e24   # P-side doping (acceptors)
    ND = 1e22   # N-side doping (donors)
    
    # Built-in potential
    Vbi = Vt * np.log(NA * ND / 1e32)
    
    # Depletion widths (1D junction assumption)
    eps = eps_si
    W = np.sqrt(2 * eps / q * (NA + ND) / (NA * ND) * (Vbi - Vbias))
    xp = -W * ND / (NA + ND)
    xn = W * NA / (NA + ND)
    
    # Spatial grid
    x = np.linspace(-1e-6, 1e-6, 1000)
    E = np.zeros_like(x)
    V = np.zeros_like(x)
    
    # Electric field and potential
    for i, xi in enumerate(x):
        if xi < xp:
            E[i] = 0
        elif xp <= xi <= 0:
            E[i] = -q * NA * (xi - xp) / eps
        elif 0 < xi <= xn:
            E[i] = q * ND * (xn - xi) / eps
        else:
            E[i] = 0
    
    # Integrate E to get potential (energy bands)
    V = -np.cumsum(E) * (x[1] - x[0])
    Ec = V.copy()
    Ev = V - 1.12  # Eg ~ 1.12 eV for Si

    # Carrier concentrations (assuming thermal equilibrium)
    ni = 1.5e16  # Intrinsic carrier concentration for Si (1/m^3)

    # Initialize arrays for electron and hole concentration
    n = np.zeros_like(x)  # Electron concentration
    p = np.zeros_like(x)  # Hole concentration
    
    # Electrostatic potential (phi)
    phi = V / q  # Convert energy potential to electrostatic potential
    
    # Electron and hole concentration (from Fermi-Dirac approx)
    n = ni * np.exp((phi - Vbi) / Vt)
    p = ni * np.exp(-(phi - Vbi) / Vt)
    
    return x, Ec, Ev, n, p, E, V

# Function to create the plots
def create_plots(x, Ec, Ev, n, p, E, V, Vbias):
    # Plot 1: Energy Band Diagram
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x * 1e6, Ec, label="Conduction Band (Ec)", color="blue")
    ax1.plot(x * 1e6, Ev, label="Valence Band (Ev)", color="red")
    ax1.axvline(x=-np.sqrt(2 * eps_si * q * (1e24 + 1e22) / (1e24 * 1e22) * (Vt * np.log(1e24 * 1e22 / 1e32) - Vbias)) * 1e6, color='gray', linestyle='--', label="Depletion Edge")
    ax1.axvline(x=np.sqrt(2 * eps_si * q * (1e24 + 1e22) / (1e24 * 1e22) * (Vt * np.log(1e24 * 1e22 / 1e32) - Vbias)) * 1e6, color='gray', linestyle='--')
    ax1.set_title(f'PN Junction Energy Band Diagram\nBias Voltage = {Vbias} V')
    ax1.set_xlabel("Position (μm)")
    ax1.set_ylabel("Energy (eV)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Carrier Concentrations
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.semilogy(x * 1e6, n, label="Electron Concentration (n)", color='blue')
    ax2.semilogy(x * 1e6, p, label="Hole Concentration (p)", color='red')
    ax2.axvline(x=-np.sqrt(2 * eps_si * q * (1e24 + 1e22) / (1e24 * 1e22) * (Vt * np.log(1e24 * 1e22 / 1e32) - Vbias)) * 1e6, color='gray', linestyle='--', label="Depletion Edge")
    ax2.axvline(x=np.sqrt(2 * eps_si * q * (1e24 + 1e22) / (1e24 * 1e22) * (Vt * np.log(1e24 * 1e22 / 1e32) - Vbias)) * 1e6, color='gray', linestyle='--')
    ax2.set_title(f'Carrier Concentration Across PN Junction\nBias = {Vbias} V')
    ax2.set_xlabel("Position (μm)")
    ax2.set_ylabel("Carrier Density (log scale)")
    ax2.grid(True, which='both')
    ax2.legend()

    # Plot 3: Electric Field
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(x * 1e6, E, label="Electric Field (E)", color='green')
    ax3.axvline(x=-np.sqrt(2 * eps_si * q * (1e24 + 1e22) / (1e24 * 1e22) * (Vt * np.log(1e24 * 1e22 / 1e32) - Vbias)) * 1e6, color='gray', linestyle='--', label="Depletion Edge")
    ax3.axvline(x=np.sqrt(2 * eps_si * q * (1e24 + 1e22) / (1e24 * 1e22) * (Vt * np.log(1e24 * 1e22 / 1e32) - Vbias)) * 1e6, color='gray', linestyle='--')
    ax3.set_title(f'PN Junction Electric Field\nBias Voltage = {Vbias} V')
    ax3.set_xlabel("Position (μm)")
    ax3.set_ylabel("Electric Field (V/m)")
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Diode I-V Curve
    I = q * 1e-4 * 1.5e16**2 * (np.exp(V / Vt) - 1)  # Diode current (simplified equation)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(V, I, label="I-V Curve", color='purple')
    ax4.set_title("Diode I-V Characteristics")
    ax4.set_xlabel("Voltage (V)")
    ax4.set_ylabel("Current (A)")
    ax4.grid(True)
    ax4.legend()

    return fig1, fig2, fig3, fig4

# Streamlit App
def main():
    st.title('PN Junction Diode Simulator')

    # Sliders for bias voltage
    Vbias = st.slider('Bias Voltage (V)', -1.0, 1.0, 0.5)
    
    # Run the simulation and get the plots
    x, Ec, Ev, n, p, E, V = simulate_pn_junction(Vbias)
    
    # Create and display all plots
    fig1, fig2, fig3, fig4 = create_plots(x, Ec, Ev, n, p, E, V, Vbias)
    
    # Display plots in Streamlit
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)

if __name__ == '__main__':
    main()
