import numpy as np
import torch
from torch.func import grad, vmap
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss
import random as rd
# PINNS :

# réseau de neurones :


def initialize_parameters(dimension):
    parameters = {}
    n = len(dimension)
    for layer in range(1, n):
        parameters['W' + str(layer)] = torch.randn(dimension[layer], dimension[layer-1], dtype=torch.float64) * (1.0 / dimension[layer-1]) ** (1/2)
        parameters['b' + str(layer)] = torch.zeros((dimension[layer], 1), dtype=torch.float64)
    return parameters

def sigmoid(Z):
    return 1 / (1 + torch.exp(-torch.clip(Z, -500, 500)))

def u_network(params, x, t):
    # version torch pour torch.func.grad — miroir de forward_propagation
    L = len(params) // 2
    def to_tensor(x, dtype=torch.float64):
        if isinstance(x, torch.Tensor):
            return x.to(dtype)
    x, t = to_tensor(x), to_tensor(t)
    
    A = torch.stack([x, t]).reshape(-1, 1)
    
    for layer in range(1, L + 1):
        W = params['W' + str(layer)]
        b = params['b' + str(layer)]
        #print("W shape :", W.shape, "A shape: ", A.shape)
       
        Z = W @ A + b
        A = Z if layer == L else torch.tanh(Z)
    return A.squeeze()

def edp(u, x, t, alpha=0.01):
    du_dt  = grad(u, argnums=1)(x, t)       # ∂u/∂t
    du_dx  = grad(u, argnums=0)       # ∂u/∂x
    du_dxx = grad(du_dx, argnums=0)(x, t)    # ∂²u/∂x²
    return du_dt - alpha * du_dxx 

def residual(params, x, t, alpha=0.01):
    return edp(lambda x, t: u_network(params, x, t), x, t, alpha)

def loss_physics(params, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc):

    r = vmap(lambda x, t: residual(params, x, t))(x_col, t_col)
    loss_res = torch.mean(r**2)

    u_pred_ic = vmap(lambda x, t: u_network(params, x, t))(x_ic, t_ic)
    loss_ci   = torch.mean((u_pred_ic - u_ic)**2)

    u_pred_bc = vmap(lambda x, t: u_network(params, x, t))(x_bc, t_bc)
    loss_cl   = torch.mean((u_pred_bc - u_bc)**2)
    
    return loss_res + loss_ci + loss_cl



def update_parameters(parameters, learning_rate):
    L = len(parameters) // 2
    new_params = {}

    Loss = loss_physics  # à calculer
    grads = grad(Loss,argnums=0)(parameters, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc) # complete with torch's grad function  
    for layer in range(1, L+1):
        new_params['W' + str(layer)] = parameters['W' + str(layer)] - learning_rate * grads['W' + str(layer)]
        new_params['b' + str(layer)] = parameters['b' + str(layer)] - learning_rate * grads['b' + str(layer)]
    return new_params

# entraînement :

def train_neural_network(dimension, n_iterations, learning_rate=0.1, log_interval=10):
    parameters = initialize_parameters(dimension)
    history = {'loss': []}
    snapshots = []   # (iteration, loss)

    for iteration in tqdm(range(n_iterations), desc="Training"):
        parameters = update_parameters(parameters, learning_rate)
        if iteration % log_interval == 0:
            loss = loss_physics(parameters, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc)
            history['loss'].append(loss.item())
            snapshots.append((iteration, loss.item()))   # juste iteration + loss
        if iteration >= 1200:
            learning_rate = 0.01

    return parameters, history, snapshots

# --- Entraînement ---
t = torch.linspace(0, 1, 100)


# --- Domaine : x ∈ [0,1], t ∈ [0,1] ---
N_col, N_ic, N_bc = 1000, 100, 100

# points de collocation (résidu EDP, intérieur du domaine)
x_col = torch.rand(N_col, dtype=torch.float64)
t_col = torch.rand(N_col, dtype=torch.float64)

# conditions initiales : t = 0, u(x, 0) = à définir
x_ic  = torch.rand(N_ic, dtype=torch.float64)
t_ic  = torch.zeros(N_ic, dtype=torch.float64)
u_ic  = torch.zeros(N_ic, dtype=torch.float64)  # à compléter selon l'EDP

# conditions aux limites : x = 0 et x = 1, u = 0 (Dirichlet)
x_bc  = torch.cat([torch.zeros(N_bc // 2), torch.ones(N_bc // 2)]).to(torch.float64)
t_bc  = torch.rand(N_bc, dtype=torch.float64)
u_bc  = torch.zeros(N_bc, dtype=torch.float64)

parameters, history, snapshots = train_neural_network(
    dimension=[2, 8, 1],
    n_iterations=4000,
    learning_rate=0.05,
    log_interval=10
)

"""iterations_log = [s[0] for s in snapshots]
losses         = [s[1] for s in snapshots]   # index 1, pas 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Apprentissage de l'EDP", fontsize=13)

# -- Solution finale --
x_plot  = torch.linspace(0, 1, 200, dtype=torch.float64)
t_fixed = torch.full((200,), 0.5, dtype=torch.float64)
u_plot  = vmap(lambda x, t: u_network(parameters, x, t))(x_plot, t_fixed)

ax1.plot(x_plot.detach(), u_plot.detach(), 'b-', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('u(x, t=0.5)')
ax1.set_title('Solution apprise')
ax1.grid(True, alpha=0.3)

# -- Courbe de loss --
ax2.semilogy(iterations_log, losses, 'g-', linewidth=2)
ax2.set_xlabel('Itération')
ax2.set_ylabel('Loss')
ax2.set_title('Courbe de loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""

x_plot = torch.linspace(0, 10, 1000, dtype=torch.float64)
t_plot = torch.linspace(0, 10, 1000, dtype=torch.float64)

# grille 2D
X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
x_flat = X.flatten()
t_flat = T.flatten()

# évaluation sur toute la grille
u_flat = vmap(lambda x, t: u_network(parameters, x, t))(x_flat, t_flat)
U = u_flat.detach().reshape(1000, 1000)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.numpy(), T.numpy(), U.numpy(), cmap='viridis')
ax.set_xlabel('x'); ax.set_ylabel('t'); ax.set_zlabel('u')
plt.show()