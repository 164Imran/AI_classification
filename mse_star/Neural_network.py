import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss
import random as rd
import jax.numpy as jnp
import jax

def F(x,z):
    return np.array([np.cos(x+5*pi/6) , np.sin(z)])
pi = np.pi
x = np.linspace(0, 2*pi, 100).reshape(1, 100)
y = F(x, x)

def f(n_points=100, noise=0):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    x = 3*np.cos(t) + np.random.normal(0, noise, n_points)
    y = 3*np.sin(t) + np.random.normal(0, noise, n_points)
    
    return np.array([x, y])



def star_array(n_branches=5, R=1.0, r=0.4, n_points=100):
    """
    Retourne un array (2, n_points) de points interpolés sur la courbe d'une étoile.
    Ligne 0 = x, ligne 1 = y.
    """
    # Construire les sommets
    n_vertices = 2 * n_branches
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    angles -= np.pi / 2

    radii = np.where(np.arange(n_vertices) % 2 == 0, R, r)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    # Fermer le polygone
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Paramètre cumulatif (distance le long du polygone)
    t_vertices = np.zeros(len(x))
    for i in range(1, len(x)):
        t_vertices[i] = t_vertices[i-1] + np.hypot(x[i] - x[i-1], y[i] - y[i-1])
    t_vertices /= t_vertices[-1]  # normaliser entre 0 et 1

    # Interpoler sur n_points réguliers
    t = np.linspace(0, 1, n_points, endpoint=False)
    x_interp = np.interp(t, t_vertices, x)
    y_interp = np.interp(t, t_vertices, y)

    return np.array([x_interp, y_interp])
# réseau de neurones :

def initialize_parameters(dimension):
    np.random.seed(42)
    parameters = {}
    n = len(dimension)
    for layer in range(1, n):
        parameters['W' + str(layer)] = np.random.randn(dimension[layer], dimension[layer-1]) * np.sqrt(1.0 / dimension[layer-1])
        parameters['b' + str(layer)] = np.zeros((dimension[layer], 1))
    return parameters


def sigmoid(Z):
    return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # clip pour éviter overflow


def forward_propagation(X, parameters):
    L = len(parameters) // 2
    cache = {'A0': X}
    for layer in range(1, L + 1):
        W = parameters['W' + str(layer)]
        Z = W.dot(cache['A' + str(layer-1)]) + parameters['b' + str(layer)]
        cache['Z' + str(layer)] = Z
        if layer == L:
            cache['A' + str(layer)] = Z
        else:   
            cache['A' + str(layer)] = np.tanh(Z)
    return cache


def backward_propagation(X, y, parameters, cache):
    L = len(parameters) // 2
    m = y.shape[1]
    gradients = {}

    dZ = cache['A' + str(L)] - y   # erreur de sortie

    for layer in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(layer-1)]
        gradients['dW' + str(layer)] = (1/m) * dZ.dot(A_prev.T)
        gradients['db' + str(layer)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            W = parameters['W' + str(layer)]
            A_prev_activation = cache['A' + str(layer-1)]
            # dZ = np.dot(W.T, dZ) * A_prev_activation * (1 - A_prev_activation) derivé de la sigmoid
            dZ = np.dot(W.T, dZ) * (1 - A_prev_activation ** 2)  # tanh

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for layer in range(1, L+1):
        parameters['W' + str(layer)] -= learning_rate * gradients['dW' + str(layer)]
        parameters['b' + str(layer)] -= learning_rate * gradients['db' + str(layer)]
    return parameters


def predict(X, parameters, threshold=0.5):
    cache = forward_propagation(X, parameters)
    A = cache['A' + str(len(parameters) // 2)]
    return A >= threshold


# entraînement :

def train_neural_network(X, y, dimension, n_iterations, learning_rate=0.1, log_interval=10):
    parameters = initialize_parameters(dimension)
    history = {'loss': []}
    snapshots = []  # (iteration, y_pred, loss)
    for iteration in tqdm(range(n_iterations), desc="Training"):
        cache = forward_propagation(X, parameters)
        gradients = backward_propagation(X, y, parameters, cache)
        parameters = update_parameters(parameters, gradients, learning_rate)
        if iteration % log_interval == 0:
            A_out = cache['A' + str(len(dimension)-1)]
            loss = np.mean((A_out - y) ** 2)
            history['loss'].append(loss)
            snapshots.append((iteration, A_out.copy(), loss))
        if iteration >= 1200:
            learning_rate = 0.01
    return parameters, history, snapshots

# --- Entraînement ---
print(star_array().shape)
t = np.linspace(0, 2 * np.pi, 100, endpoint=False)

# Cercle = entrée
X = np.array([3 * np.cos(t), 3 * np.sin(t)]) 
y = star_array()
print(y.shape, X.shape)

parameters, history, snapshots = train_neural_network(
    X, y, dimension=[2, 32, 2], n_iterations=15000, learning_rate=0.05, log_interval=10
)

# --- Animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Apprentissage de F(x) = 2x - 1', fontsize=13)

# Axe gauche : courbe apprise vs cible
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Prédiction du réseau')
ax1.grid(True, alpha=0.3)
ax1.plot(y[0], y[1], 'r--', linewidth=2, label='F(x) cible')
line_pred, = ax1.plot([], [], 'b-', linewidth=2, label='Prédiction')
ax1.legend()

# Axe droite : courbe de loss
iterations_log = [s[0] for s in snapshots]
losses = [s[2] for s in snapshots]
ax2.set_xlim(0, max(iterations_log))
ax2.set_ylim(0, max(losses) * 1.1)
ax2.set_xlabel('Itération')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Courbe de loss')
ax2.grid(True, alpha=0.3)
line_loss, = ax2.plot([], [], 'g-', linewidth=2)
loss_dot, = ax2.plot([], [], 'go', markersize=7)
iter_text = ax1.text(0.02, 1.35, '', fontsize=10, color='gray')
from matplotlib import animation
def update(frame):
    iteration, y_pred, loss = snapshots[frame]
    line_pred.set_data(y_pred[0], y_pred[1])
    line_loss.set_data(iterations_log[:frame+1], losses[:frame+1])
    loss_dot.set_data([iterations_log[frame]], [losses[frame]])
    iter_text.set_text(f'Itération {iteration} — Loss: {loss:.4f}')
    return line_pred, line_loss, loss_dot, iter_text

ani = animation.FuncAnimation(
    fig, update, frames=len(snapshots), interval=20, blit=True)
plt.plot(X[0], X[1])
plt.tight_layout()
plt.show()
