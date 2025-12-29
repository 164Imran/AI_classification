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

def load_data():
    """
    Load training and test datasets from HDF5 files.

    Returns
    -------
    X_train : ndarray, shape (n_samples, height, width, channels)
        Training images
    y_train : ndarray, shape (n_samples,)
        Training labels (0 or 1)
    X_test : ndarray, shape (n_samples, height, width, channels)
        Test images
    y_test : ndarray, shape (n_samples,)
        Test labels (0 or 1)
    """
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:])
    y_train = np.array(train_dataset["Y_train"][:])

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:])
    y_test = np.array(test_dataset["Y_test"][:])

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

def initialize_parameters(n_input, n_hidden, n_output):
    """
    Initialize weights and biases for a 2-layer neural network.

    Parameters
    ----------
    n_input : int
        Number of input features (n0)
    n_hidden : int
        Number of neurons in hidden layer (n1)
    n_output : int
        Number of output neurons (n2, should be 1 for binary classification)

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

    # Hidden layer parameters
    W1 = np.random.randn(n_hidden, n_input) * 0.01  # Small initialization
    b1 = np.zeros((n_hidden, 1))  # Better than random

    # Output layer parameters
    W2 = np.random.randn(n_output, n_hidden) * 0.01
    b2 = np.zeros((n_output, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

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

    Parameters
    ----------
    X : ndarray, shape (n_features, n_samples)
        Input data
    parameters : dict
        Network parameters (W1, b1, W2, b2)

    Returns
    -------
    cache : dict
        Dictionary containing:
        - A1 : Hidden layer activations, shape (n_hidden, n_samples)
        - A2 : Output layer activations, shape (1, n_samples)

    Notes
    -----
    Architecture:
        Input (X) -> [W1, b1] -> Sigmoid -> A1 -> [W2, b2] -> Sigmoid -> A2
    """
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']

    # Hidden layer
    Z1 = W1.dot(X) + b1  # Linear combination
    A1 = sigmoid(Z1)  # Activation

    # Output layer
    Z2 = W2.dot(A1) + b2  # Linear combination
    A2 = sigmoid(Z2)  # Activation (probability)

    cache = {
        'A1': A1,
        'A2': A2
    }

    return cache


def backward_propagation(X, y, parameters, cache):
    """
    Compute gradients using backpropagation algorithm.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_samples)
        Input data
    y : ndarray, shape (1, n_samples)
        True labels
    parameters : dict
        Network parameters
    cache : dict
        Forward propagation cache (A1, A2)

    Returns
    -------
    gradients : dict
        Dictionary containing gradients:
        - dW1, dW2 : Weight gradients
        - db1, db2 : Bias gradients

    Notes
    -----
    Uses chain rule to compute partial derivatives of loss w.r.t. parameters.
    Loss function: Binary cross-entropy
    """
    A1, A2 = cache['A1'], cache['A2']
    W2 = parameters['W2']

    m = y.shape[1]  # Number of samples

    # Output layer gradients
    dZ2 = A2 - y  # Derivative of loss w.r.t. Z2 (cross-entropy + sigmoid)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer gradients
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)  # Chain rule with sigmoid derivative
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1': dW1,
        'dW2': dW2,
        'db1': db1,
        'db2': db2
    }

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
    W1 = parameters['W1'] - learning_rate * gradients['dW1']
    b1 = parameters['b1'] - learning_rate * gradients['db1']
    W2 = parameters['W2'] - learning_rate * gradients['dW2']
    b2 = parameters['b2'] - learning_rate * gradients['db2']

    updated_parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return updated_parameters


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
    A2 = cache['A2']

    predictions = A2 >= threshold

    return predictions


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_neural_network(X, y, n_hidden, n_iterations, learning_rate=0.1,
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
    n_input = X.shape[0]
    n_output = y.shape[0]

    # Initialize parameters
    parameters = initialize_parameters(n_input, n_hidden, n_output)

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
            loss = log_loss(y.flatten(), cache['A2'].flatten())
            accuracy = accuracy_score(y.flatten(), y_pred.flatten())

            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

    return parameters, history


def plot_training_history(history, log_interval=10):
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

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

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()

    # Preprocess
    print("Preprocessing data...")
    X_train, y_train, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test
    )

    print(f"Training set: {X_train.shape[1]} samples, {X_train.shape[0]} features")
    print(f"Test set: {X_test.shape[1]} samples")

    # Train model
    print("\nTraining neural network...")
    parameters, history = train_neural_network(
        X=X_train,
        y=y_train,
        n_hidden=32,  # Number of hidden neurons
        n_iterations=1000,
        learning_rate=0.1,
        log_interval=10
    )

    # Plot training curves
    plot_training_history(history, log_interval=10)

    # Evaluate on training set
    y_train_pred = predict(X_train, parameters)
    train_accuracy = accuracy_score(y_train.flatten(), y_train_pred.flatten())
    print(f"\nTraining accuracy: {train_accuracy:.4f}")

    # Evaluate on test set
    y_test_pred = predict(X_test, parameters)
    test_accuracy = accuracy_score(y_test.flatten(), y_test_pred.flatten())
    print(f"Test accuracy: {test_accuracy:.4f}")
