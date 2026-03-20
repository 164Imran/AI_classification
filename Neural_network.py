"""
Neural Network Implementation - Binary Classification
======================================================
A simple 2-layer neural network for binary classification tasks.

Architecture:
- Input layer: n0 neurons (feature dimension)
- Hidden layer: n1 neurons (sigmoid activation)
- Output layer: 1 neuron (sigmoid activation)

Training: Gradient descent with backpropagation
Loss: Binary cross-entropy (log loss)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss
import h5py


# =============================================================================
# DATA LOADING
# =============================================================================

from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist_binary():
    """
    Charge MNIST et garde seulement les chiffres 0 et 1.
    
    Returns
    -------
    X_train, y_train, X_test, y_test : ndarrays
        Train: ~13,000 exemples
        Test: ~2,000 exemples
    """
    print("Téléchargement MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist['data'], mnist['target'].astype(int)
    X = mnist['data'].to_numpy()  # ← AJOUTEZ .to_numpy()
    y = mnist['target'].astype(int).to_numpy()  # ← AJOUTEZ .to_numpy()
    # Garder seulement 0 et 1
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    
    # Normaliser
    X = X / 255.0
    
    # Split train/test (80/20)
    n = len(X)
    n_train = int(0.8 * n)
    
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train = X[train_idx].T  # Shape: (784, n_train)
    y_train = y[train_idx].reshape(1, -1)
    X_test = X[test_idx].T
    y_test = y[test_idx].reshape(1, -1)
    
    print(f"Train: {X_train.shape[1]} exemples")
    print(f"Test: {X_test.shape[1]} exemples")
    print(f"Features: {X_train.shape[0]}")
    
    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test):
    """
    Flatten and normalize image data.

    Parameters
    ----------
    X_train, X_test : ndarray
        Image arrays with shape (n_samples, height, width, channels)
    y_train, y_test : ndarray
        Label arrays with shape (n_samples,)

    Returns
    -------
    X_train_flat : ndarray, shape (n_features, n_samples)
        Flattened and normalized training data (transposed)
    y_train_reshaped : ndarray, shape (1, n_samples)
        Reshaped training labels
    X_test_flat : ndarray, shape (n_features, n_samples)
        Flattened and normalized test data (transposed)
    y_test_reshaped : ndarray, shape (1, n_samples)
        Reshaped test labels
    """
    # Flatten images: (n_samples, h, w, c) -> (n_samples, h*w*c)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Normalize to [0, 1] using global maximum
    max_value = max(np.max(X_train_flat), np.max(X_test_flat))
    X_train_flat = X_train_flat / max_value
    X_test_flat = X_test_flat / max_value

    # Transpose to (n_features, n_samples) for matrix operations
    X_train_flat = X_train_flat.T
    X_test_flat = X_test_flat.T

    # Reshape labels to (1, n_samples)
    y_train_reshaped = y_train.reshape(1, -1)
    y_test_reshaped = y_test.reshape(1, -1)

    return X_train_flat, y_train_reshaped, X_test_flat, y_test_reshaped


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

def initialize_parameters(dimension):
    """
    Initialize weights and biases for a 2-layer neural network.

    Parameters
    ----------
    
    Returns
    -------
    parameters : dict
        Dictionary containing:
        - W1 : ndarray, shape (n_hidden, n_input)
        - b1 : ndarray, shape (n_hidden, 1)
        - W2 : ndarray, shape (n_output, n_hidden)
        - b2 : ndarray, shape (n_output, 1)

    Notes
    -----
    Weights initialized with random normal distribution.
    Biases initialized to zeros would be better practice.
    """
    np.random.seed(42)  # For reproducibility
    parameters = {}
    n = len(dimension)
    for layer in range(1, n):
        parameters['W' + str(layer)] = np.random.randn(dimension[layer],dimension[layer-1])  * np.sqrt(1.0 / dimension[layer-1])
        parameters['b' + str(layer)] = np.zeros((dimension[layer], 1))  

    return parameters


def sigmoid(Z):
    """
    Compute sigmoid activation function.

    Parameters
    ----------
    Z : ndarray
        Linear combination (pre-activation)

    Returns
    -------
    A : ndarray
        Sigmoid activation values in [0, 1]
    """
    return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to avoid overflow

def forward_propagation(X, parameters):
    """
    Perform forward propagation through the network.
    """
    # Calculer le nombre de couches (nombre de paires W/b)
    L = len(parameters) // 2  # Nombre de couches
    
    cache = {}
    cache['A0'] = X  # Stocker l'entrée aussi
    
    for layer in range(1, L + 1):
        W = parameters['W' + str(layer)]
        
        Z = W.dot(cache['A' + str(layer-1)]) + parameters['b' + str(layer)]
        cache['Z' + str(layer)] = Z  # Stocker Z pour la backprop
        cache['A' + str(layer)] = sigmoid(Z)
    
    return cache


def backward_propagation(X, y, parameters, cache):
    """
    Compute gradients using backpropagation algorithm.
    """
    L = len(parameters) // 2  # Nombre de couches
    m = y.shape[1]
    gradients = {}
    
    # Gradient de la dernière couche (erreur de sortie)
    dZ = cache['A' + str(L)] - y
    
    # Backpropagation à travers toutes les couches
    for layer in reversed(range(1, L + 1)):
        # Gradients des poids et biais
        A_prev = cache['A' + str(layer-1)]
        gradients['dW' + str(layer)] = (1/m) * dZ.dot(A_prev.T)
        gradients['db' + str(layer)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Propager le gradient à la couche précédente (sauf pour la première)
        if layer > 1:
            W = parameters['W' + str(layer)]
            A_prev_activation = cache['A' + str(layer-1)]
            dZ = np.dot(W.T, dZ) * A_prev_activation * (1 - A_prev_activation)
    
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    """
    Update parameters using gradient descent.

    Parameters
    ----------
    parameters : dict
        Current network parameters
    gradients : dict
        Computed gradients
    learning_rate : float
        Step size for gradient descent

    Returns
    -------
    parameters : dict
        Updated parameters
    """
    L = len(parameters) // 2
    update = {}
    for layer in range(1, L+1):
        
        parameters['W' + str(layer)] -= learning_rate * gradients['dW' + str(layer)]
        parameters['b' + str(layer)] -= learning_rate * gradients['db' + str(layer)]

    return parameters


def predict(X, parameters, threshold=0.5):
    """
    Make binary predictions using trained network.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_samples)
        Input data
    parameters : dict
        Trained network parameters
    threshold : float, default=0.5
        Classification threshold

    Returns
    -------
    predictions : ndarray, shape (1, n_samples)
        Binary predictions (True/False or 1/0)
    """
    cache = forward_propagation(X, parameters)
    A = cache['A' + str(len(parameters) // 2 )]

    predictions = A >= threshold
    
    return predictions


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_neural_network(X, y, dimension, n_iterations, learning_rate=0.1,
                         log_interval=10):
    """
    Train a 2-layer neural network using gradient descent.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_samples)
        Training data
    y : ndarray, shape (1, n_samples)
        Training labels
    n_hidden : int
        Number of neurons in hidden layer
    n_iterations : int
        Number of training iterations
    learning_rate : float, default=0.1
        Learning rate for gradient descent
    log_interval : int, default=10
        Compute metrics every `log_interval` iterations

    Returns
    -------
    parameters : dict
        Trained network parameters
    history : dict
        Training history containing:
        - loss : List of loss values
        - accuracy : List of accuracy values
    """
    # Initialize parameters
    parameters = initialize_parameters(dimension)

    # Training history
    history = {
        'loss': [],
        'accuracy': []
    }

    # Training loop
    for iteration in tqdm(range(n_iterations), desc="Training"):
        # Forward propagation
        cache = forward_propagation(X, parameters)

        # Backward propagation
        gradients = backward_propagation(X, y, parameters, cache)

        # Update parameters
        parameters = update_parameters(parameters, gradients, learning_rate)

        # Log metrics
        if iteration % log_interval == 0:
            # Predictions
            y_pred = predict(X, parameters)
            # Metrics
            loss = log_loss(y.flatten(), cache['A'+str(len(dimension)-1)].flatten())
            
            accuracy = accuracy_score(y.flatten(), y_pred.flatten())

            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

    return parameters, history, X, y


def plot_training_history(X, y, history, log_interval=10):
    """
    Plot training loss and accuracy curves.

    Parameters
    ----------
    history : dict
        Training history from train_neural_network
    log_interval : int
        Interval used during training
    """
    iterations = np.arange(len(history['loss'])) * log_interval

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Loss curve
    ax1.plot(iterations, history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (Binary Cross-Entropy)')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(iterations, history['accuracy'], 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)


    # Accuracy curve
    ax3.scatter(X[:, 0], X[:, 1], c= y, s=20)
    ax3.set_title('Data Visualisation')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Utilisation
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_binary()
    
    # Train
    parameters, history, X, y = train_neural_network(
        X=X_train,
        y=y_train,
        dimension=[784, 64, 16, 1],
        n_iterations=1000,
        learning_rate=0.1,
        log_interval=100
    )
    
    # Evaluate
    y_train_pred = predict(X_train, parameters)
    y_test_pred = predict(X_test, parameters)
    
    train_acc = accuracy_score(y_train.flatten(), y_train_pred.flatten())
    test_acc = accuracy_score(y_test.flatten(), y_test_pred.flatten())
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    plot_training_history(X, y, history, log_interval=10)

