
import numpy as np
import matplotlib.pyplot as plt

#constantes :

Cm  = 1.0    # capacité membranaire (µF/cm²)
g_Na = 120.0  # conductance max Na  (mS/cm²)
g_K  = 36.0   # conductance max K   (mS/cm²)
g_L  = 0.3    # conductance fuite   (mS/cm²)
E_Na = 50.0   # potentiel équilibre Na  (mV)
E_k  = -77.0  # potentiel équilibre K   (mV)
E_l  = -54.387# potentiel équilibre fuite (mV)
V0=-65.00
n0=0.3177
m0=0.0530
h0=0.5961

alpha_m = lambda V: 0.1*(V+40) / (1 - np.exp(-(V+40)/10) + 1e-9)
beta_m  = lambda V: 4.0 * np.exp(-(V+65)/18)

alpha_h = lambda V: 0.07 * np.exp(-(V+65)/20)
beta_h  = lambda V: 1.0 / (1 + np.exp(-(V+35)/10))

alpha_n = lambda V: 0.01*(V+55) / (1 - np.exp(-(V+55)/10) + 1e-9)
beta_n  = lambda V: 0.125 * np.exp(-(V+65)/80)
    
def derivative(state):

    V, n, m, h= state[0], state[1], state[2], state[3]
    dV = - (g_Na * m**3 * h * (V - E_Na) + g_K * n**4 * (V - E_k) + g_L * (V - E_l) - I_ext)/Cm
    dn = alpha_n(V)*(1-n) -beta_n(V)*n
    dm = alpha_m(V)*(1-m) -beta_m(V)*m
    dh = alpha_h(V)*(1-h) -beta_h(V)*h
    
    return np.array([dV, dn, dm, dh])

def rk4(f, y, t, dt):

    k1 = f(y)
    k2 = f(y + dt/2 * k1)
    k3 = f(y + dt/2 * k2)
    k4 = f(y + dt   * k3)

    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def simulate(I_ext, T=50, dt=0.01):
    time = np.arange(0, T, dt)
    V = np.zeros_like(time)
    n = np.zeros_like(time)
    m = np.zeros_like(time)
    h = np.zeros_like(time)

    # conditions initiales
    V[0], n[0], m[0], h[0] = V0, n0, m0, h0

    for i in range(1, len(time)):
        
        state = np.array([V[i-1], n[i-1], m[i-1], h[i-1]])
        V[i], n[i], m[i], h[i] = rk4(derivative, state, time[i-1], dt)

    return time, V, n, m, h

# Simulation
fig, ax = plt.subplots()
plt.ion()

for i_val in np.linspace(0, 50, 100):
    I_ext = i_val  # scalaire utilisé par derivative() via la portée globale
    t, V, n, m, h = simulate(I_ext, T=100, dt=0.01)

    ax.clear()
    ax.plot(t, V * 0.05, c='purple', label='V×0.05')
    ax.plot(t, n, c='g', label='n')
    ax.plot(t, m, c='r', label='m')
    ax.plot(t, h, c='b', label='h')
    ax.set_title(f'Hodgkin-Huxley — I_ext = {i_val:.1f} µA/cm²')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV) / Gating Variables')
    ax.legend()
    ax.grid()
    plt.pause(0.05)

plt.ioff()
plt.show()