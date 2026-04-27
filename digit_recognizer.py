import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


# ── Primitives CNN ─────────────────────────────────────────────────────────────

def conv(Z, kernel):
    n_Z, n_K = len(Z), len(kernel)
    n = n_K // 2
    Z_padded = np.pad(Z, n, mode='constant')
    Y = np.zeros(Z.shape, dtype=np.float64)
    for i in range(n_Z):
        for j in range(n_Z):
            Y[i, j] = np.sum(Z_padded[i:i+n_K, j:j+n_K] * kernel)
    return Y


def ReLU(x):
    return np.maximum(x, 0)


def d_ReLU(X):
    return (X > 0).astype(float)


def max_pooling(Z, n):
    n_Z = len(Z)
    max_pool = []
    for i in range(0, n_Z, n):
        for j in range(0, n_Z, n):
            max_pool.append(np.max(Z[i:i+n, j:j+n]))
    return np.array(max_pool)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def cross_entropy_loss(A, y):
    m = y.shape[1]
    A_clipped = np.clip(A, 1e-15, 1 - 1e-15)
    return -(1 / m) * np.sum(y * np.log(A_clipped))


def compute_dK_batch(X_batch, dZ_conv_batch, kernel_size):
    dK = np.zeros((kernel_size, kernel_size))
    n = kernel_size // 2
    m = X_batch.shape[0]
    for b in range(m):
        img_padded = np.pad(X_batch[b], n, mode='constant')
        dz = dZ_conv_batch[:, :, b]
        for i in range(kernel_size):
            for j in range(kernel_size):
                dK[i, j] += np.sum(img_padded[i:i+28, j:j+28] * dz)
    return dK / m


# ── Forward / Backward / Update ────────────────────────────────────────────────

def forward_propagation(X_image, parameters, n_layers, kernel):
    cache = {}
    A0_list, Z_c_list = [], []

    for i in range(X_image.shape[0]):
        Z_c = conv(X_image[i], kernel)
        Z_c_list.append(Z_c)
        pooled = max_pooling(ReLU(Z_c), 2).reshape(-1, 1)
        A0_list.append(pooled)

    cache['A0'] = np.hstack(A0_list)
    cache['Z_c'] = np.stack(Z_c_list, axis=-1)

    A_prev = cache['A0']
    for l in range(1, len(n_layers)):
        Z = np.dot(parameters[f'W{l}'], A_prev) + parameters[f'b{l}']
        A = softmax(Z) if l == len(n_layers) - 1 else ReLU(Z)
        cache[f'Z{l}'], cache[f'A{l}'] = Z, A
        A_prev = A

    return A, cache


def backward_propagation(X, y, parameters, cache, n_layers):
    m = y.shape[1]
    grads = {}
    L = len(n_layers) - 1

    dZ = cache[f'A{L}'] - y
    for l in reversed(range(1, L + 1)):
        grads[f'dW{l}'] = (1 / m) * np.dot(dZ, cache[f'A{l-1}'].T)
        grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(parameters[f'W{l}'].T, dZ)
        if l > 1:
            dZ = dA_prev * d_ReLU(cache[f'Z{l-1}'])
        else:
            dA0 = dA_prev

    dA0_spatial = dA0.reshape(14, 14, m)
    dZ_conv = np.repeat(np.repeat(dA0_spatial, 2, axis=0), 2, axis=1)
    dZ_conv = dZ_conv * d_ReLU(cache['Z_c'])
    grads['dK'] = compute_dK_batch(X, dZ_conv, kernel_size=3)

    return grads


def updates(parameters, gradient, kernel, learning_rate, lambd, n_layers):
    for i in range(1, len(n_layers)):
        parameters[f'W{i}'] -= learning_rate * (gradient[f'dW{i}'] + lambd * parameters[f'W{i}'])
        parameters[f'b{i}'] -= learning_rate * gradient[f'db{i}']
    kernel -= learning_rate * gradient['dK']
    return parameters, kernel


# ── Utilitaires ────────────────────────────────────────────────────────────────

def init(N: list):
    parameters = {}
    for i in range(1, len(N)):
        parameters[f'W{i}'] = np.random.randn(N[i], N[i-1]) * np.sqrt(2 / N[i-1])
        parameters[f'b{i}'] = np.zeros((N[i], 1))
    return parameters


def one_hot(y_labels):
    m = len(y_labels)
    encoded = np.zeros((10, m))
    for i, val in enumerate(y_labels):
        encoded[int(val), i] = 1
    return encoded


def get_accuracy(A, y):
    return np.mean(np.argmax(A, axis=0) == np.argmax(y, axis=0))


def predict(X, parameters, n_layers, kernel):
    """Retourne les classes prédites (entiers 0-9) pour un tableau d'images.

    X peut être :
      - (n, 784) en valeurs brutes 0-255
      - (784,)   image unique en valeurs brutes 0-255
    """
    if X.ndim == 2 and X.shape[1] == 784:
        X = (X / 255.0).reshape(-1, 28, 28)
    elif X.ndim == 1:
        X = (X / 255.0).reshape(1, 28, 28)
    A, _ = forward_propagation(X, parameters, n_layers, kernel)
    return np.argmax(A, axis=0)


def export_predictions(predictions, filename):
    """Sauvegarde les prédictions dans un CSV (colonnes : ImageId, Label)."""
    df = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions) + 1),
        "Label": predictions,
    })
    df.to_csv(filename, index=False)
    print(f"Prédictions sauvegardées : {filename}")


# ── Script principal ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN from-scratch — classification de chiffres MNIST")
    parser.add_argument("--train",  default="train.csv",        help="CSV d'entraînement (1ère colonne = label)")
    parser.add_argument("--test",   default="test.csv",         help="CSV de test (sans colonne label)")
    parser.add_argument("--output", default="predictions.csv",  help="Fichier CSV de sortie")
    parser.add_argument("--lr",     type=float, default=0.01,   help="Learning rate")
    parser.add_argument("--lambd",  type=float, default=0.006,  help="Coefficient L2")
    parser.add_argument("--batch",  type=int,   default=32,     help="Taille du batch")
    parser.add_argument("--seed",   type=int,   default=42,     help="Graine aléatoire")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Chargement
    df      = pd.read_csv(args.train).to_numpy()
    df_test = pd.read_csv(args.test).to_numpy()
    np.random.shuffle(df)

    X_train_raw = df[:, 1:]          # (n, 784), valeurs 0-255
    y_train     = df[:, 0]
    X_val_raw   = df[-1000:, 1:]     # (1000, 784), valeurs 0-255
    y_val       = df[-1000:, 0]

    # Architecture et initialisation
    N          = [196, 128, 64, 64, 10]
    n_batch    = args.batch
    parameters = init(N)
    kernel     = np.random.randn(3, 3) * np.sqrt(2 / 9)
    losses     = []

    # Entraînement (1 epoch)
    for step in tqdm(range(0, len(X_train_raw), n_batch), desc="Entraînement"):
        X_batch_flat = X_train_raw[step:step + n_batch]
        if X_batch_flat.shape[0] < n_batch:
            break
        X_batch = (X_batch_flat / 255.0).reshape(n_batch, 28, 28)
        y_batch = one_hot(y_train[step:step + n_batch]).reshape(10, n_batch)

        A, cache   = forward_propagation(X_batch, parameters, N, kernel)
        gradient   = backward_propagation(X_batch, y_batch, parameters, cache, N)
        parameters, kernel = updates(parameters, gradient, kernel, args.lr, args.lambd, N)
        losses.append(cross_entropy_loss(A, y_batch))

    # Accuracy sur le jeu de validation
    X_val_norm = (X_val_raw / 255.0).reshape(-1, 28, 28)
    A_val, _   = forward_propagation(X_val_norm, parameters, N, kernel)
    y_val_oh   = one_hot(y_val)
    print(f"Accuracy validation : {get_accuracy(A_val, y_val_oh):.2%}")

    # Courbe de loss
    plt.plot(losses)
    plt.xlabel("Batch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Courbe d'entraînement")
    plt.tight_layout()
    plt.show()

    # Prédictions et export
    preds = predict(df_test, parameters, N, kernel)
    export_predictions(preds, args.output)
