import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss

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
        cache['A' + str(layer)] = sigmoid(Z)
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
            dZ = np.dot(W.T, dZ) * A_prev_activation * (1 - A_prev_activation)

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
    history = {'loss': [], 'accuracy': []}

    for iteration in tqdm(range(n_iterations), desc="Training"):
        cache      = forward_propagation(X, parameters)
        gradients  = backward_propagation(X, y, parameters, cache)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if iteration % log_interval == 0:
            y_pred   = predict(X, parameters)
            loss     = log_loss(y.flatten(), cache['A'+str(len(dimension)-1)].flatten())
            accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

    return parameters, history


def plot_training_history(history, log_interval=10):
    iterations = np.arange(len(history['loss'])) * log_interval
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.plot(iterations, history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (Binary Cross-Entropy)')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(iterations, history['accuracy'], 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
