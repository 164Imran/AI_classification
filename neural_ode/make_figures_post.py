"""Génère les 3 figures LinkedIn à partir du pipeline de NODE.py.

- single_step_accuracy.png : prédiction single-step vs vérité sur states_test
- rollout_divergence.png   : rollout autorégressif vs vérité (divergence)
- attractor_3d_comparison.png : attracteur réel vs rollout en 3D

Réplique exactement les données et le modèle de NODE.py (mêmes seeds Lorenz,
mêmes stats de normalisation mu_train/sigma_train). Entraîne une fois et
sauvegarde les poids dans figures_post/node_weights.pt pour les runs suivants.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

torch.manual_seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figures_post")
os.makedirs(OUTDIR, exist_ok=True)
WEIGHTS = os.path.join(OUTDIR, "node_weights.pt")

DT = 0.01

# ---------------- Données (identiques à NODE.py) ----------------

def lorenz_deriv(state, t, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    return torch.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dim=-1)


def rk4_step(func, state, t, dt, *args):
    k1 = func(state, t, *args)
    k2 = func(state + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = func(state + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = func(state + dt * k3, t + dt, *args)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_lorenz(initial_state, t, dt):
    initial_state = torch.as_tensor(initial_state, dtype=torch.float32)
    num_steps = int(t / dt)
    states = torch.empty((num_steps + 1, *initial_state.shape), dtype=initial_state.dtype)
    states[0] = initial_state
    for i in range(num_steps):
        states[i + 1] = rk4_step(lorenz_deriv, states[i], i * dt, dt)
    return states


states = integrate_lorenz(torch.tensor([1.0, 1.0, 1.0]), t=15.0, dt=DT)
states_test = integrate_lorenz(states[-1] + torch.randn(3) * 0.1, t=200.0, dt=DT)

X_train_raw, y_train_raw = states[:-1], states[1:]

mu_train, sigma_train = X_train_raw.mean(dim=0), X_train_raw.std(dim=0)
mu_train_y, sigma_train_y = y_train_raw.mean(dim=0), y_train_raw.std(dim=0)

X_train = (X_train_raw - mu_train) / sigma_train
y_train = (y_train_raw - mu_train_y) / sigma_train_y

# ---------------- Modèle (identique à NODE.py) ----------------

class NODEMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = (torch.randn(input_size, hidden_size) * 0.1).requires_grad_()
        self.b1 = (torch.randn(1, hidden_size) * 0.1).requires_grad_()
        self.W2 = (torch.randn(hidden_size, output_size) * 0.1).requires_grad_()
        self.b2 = (torch.randn(1, output_size) * 0.1).requires_grad_()

    def forward(self, X):
        a1 = torch.sigmoid(X @ self.W1 + self.b1)
        return a1 @ self.W2 + self.b2

    def func(self, state, t):
        return self.forward(state)

    def step(self, state):
        """Un pas RK4 dans l'espace normalisé (état normalisé -> état suivant normalisé)."""
        return rk4_step(self.func, state, 0.0, DT)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]


model = NODEMLP(3, 20, 3)

if os.path.exists(WEIGHTS):
    sd = torch.load(WEIGHTS)
    with torch.no_grad():
        for p, saved in zip(model.parameters(), sd):
            p.copy_(saved)
    print("Poids chargés depuis", WEIGHTS)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in tqdm(range(15000)):
        optimizer.zero_grad()
        loss = torch.mean((model.step(X_train) - y_train) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, loss {loss.item():.3e}")
    print(f"Loss finale: {loss.item():.3e}")
    torch.save([p.detach() for p in model.parameters()], WEIGHTS)

# ---------------- Prédictions sur states_test ----------------

X_test_raw, y_test_raw = states_test[:-1], states_test[1:]

with torch.no_grad():
    # Single-step : on repart du vrai état à chaque instant
    pred_ss_norm = model.step((X_test_raw - mu_train) / sigma_train)
    pred_ss = pred_ss_norm * sigma_train_y + mu_train_y  # dénormalisation

    # Rollout autorégressif : IC donnée une seule fois
    n_roll = len(X_test_raw)
    rollout = torch.empty((n_roll + 1, 3))
    rollout[0] = X_test_raw[0]
    state = X_test_raw[0]
    for i in range(n_roll):
        state_norm = (state - mu_train) / sigma_train
        state = model.step(state_norm.unsqueeze(0)).squeeze(0) * sigma_train_y + mu_train_y
        rollout[i + 1] = state
        if not torch.isfinite(state).all():
            rollout[i + 1:] = float("nan")
            print(f"Rollout non fini au pas {i}")
            break

# Erreurs pour choisir les fenêtres honnêtement
err_ss = (pred_ss - y_test_raw).abs().max(dim=-1).values
err_roll = (rollout[1:] - y_test_raw).abs().max(dim=-1).values
print(f"Single-step | erreur max composante : médiane {err_ss.median():.4f}, max {err_ss.max():.4f}")
div_idx = int((err_roll > 1.0).float().argmax()) if (err_roll > 1.0).any() else -1
print(f"Rollout | premier pas avec erreur > 1.0 : {div_idx}")

# ---------------- Figures ----------------

FIGSIZE = (8.0, 14 / 3)  # x150 dpi = 1200x700 px
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

C_TRUE, C_PRED = "#2b2b2b", "#e4572e"

# Fenêtre commune pour les figures 1 et 2 (ajustée selon div_idx)
window = max(1000, min(3 * max(div_idx, 1), len(y_test_raw))) if div_idx > 0 else 2000
t_axis = torch.arange(window) * DT

# 1. Single-step accuracy
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(t_axis, y_test_raw[:window, 0], color=C_TRUE, lw=1.6, label="Lorenz (vérité)")
ax.plot(t_axis, pred_ss[:window, 0], color=C_PRED, lw=1.2, ls="--", label="Neural ODE — un pas à la fois")
ax.set_xlabel("temps")
ax.set_ylabel("x(t)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "single_step_accuracy.png"), dpi=150)
plt.close(fig)

# 2. Rollout divergence
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(t_axis, y_test_raw[:window, 0], color=C_TRUE, lw=1.6, label="Lorenz (vérité)")
ax.plot(t_axis, rollout[1:window + 1, 0], color=C_PRED, lw=1.2, ls="--", label="Neural ODE — rollout autorégressif")
if div_idx > 0:
    ax.axvline(div_idx * DT, color="#999999", lw=0.8, ls=":")
    ax.text(div_idx * DT, ax.get_ylim()[1], f"  divergence ≈ pas {div_idx}",
            va="top", ha="left", fontsize=9, color="#666666")
ax.set_xlabel("temps")
ax.set_ylabel("x(t)")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "rollout_divergence.png"), dpi=150)
plt.close(fig)

# 3. Attracteur 3D : vérité vs rollout
n3d = min(5000, len(y_test_raw))
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111, projection="3d")
ax.plot(*y_test_raw[:n3d].T, color=C_TRUE, lw=0.7, alpha=0.9, label="Lorenz (vérité)")
ax.plot(*rollout[1:n3d + 1].T, color=C_PRED, lw=0.7, alpha=0.9, label="Neural ODE — rollout")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.legend(frameon=False, loc="upper left")
ax.grid(False)
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "attractor_3d_comparison.png"), dpi=150)
plt.close(fig)

print("Figures écrites dans", OUTDIR)
