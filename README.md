# Neural Network - Binary Classification

A simple 2-layer neural network implementation for binary classification tasks using NumPy.

## Features

- **Architecture**: Input → Hidden (sigmoid) → Output (sigmoid)
- **Training**: Gradient descent with backpropagation
- **Loss**: Binary cross-entropy
- **Metrics**: Accuracy and loss tracking

## Installation

pip install numpy matplotlib scikit-learn h5py tqdm

text

## Usage

### Basic Example

from neural_network import train_neural_network, predict

Train model
parameters, history = train_neural_network(
X=X_train, # Shape: (n_features, n_samples)
y=y_train, # Shape: (1, n_samples)
n_hidden=32, # Number of hidden neurons
n_iterations=1000, # Training iterations
learning_rate=0.1
)

Make predictions
predictions = predict(X_test, parameters)

text

### Data Format

Expected HDF5 structure:
trainset.hdf5
├── X_train # Shape: (n_samples, height, width, channels)
└── Y_train # Shape: (n_samples,)

testset.hdf5
├── X_test
└── Y_test

text

## Architecture

Input Layer (n_input neurons)
↓
Hidden Layer (n_hidden neurons) + Sigmoid
↓
Output Layer (1 neuron) + Sigmoid
↓
Binary prediction (0 or 1)

text

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_hidden` | 32 | Number of neurons in hidden layer |
| `learning_rate` | 0.1 | Step size for gradient descent |
| `n_iterations` | 1000 | Number of training epochs |
| `threshold` | 0.5 | Classification threshold |

## Training Process

1. **Forward Propagation**: Compute predictions
2. **Loss Calculation**: Binary cross-entropy
3. **Backward Propagation**: Compute gradients
4. **Parameter Update**: Gradient descent step
5. **Repeat** for `n_iterations`

## Evaluation

from sklearn.metrics import accuracy_score, classification_report

Predictions
y_pred = predict(X_test, parameters)

Accuracy
accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
print(f"Accuracy: {accuracy:.4f}")

Detailed metrics
print(classification_report(y_test.flatten(), y_pred.flatten()))

text

## Visualization

Training curves are automatically plotted:
- **Loss curve**: Binary cross-entropy over iterations
- **Accuracy curve**: Classification accuracy over iterations

## Mathematical Details

### Forward Propagation

Z1 = W1 · X + b1
A1 = σ(Z1)
Z2 = W2 · A1 + b2
A2 = σ(Z2)

text

Where σ(z) = 1 / (1 + e^(-z)) is the sigmoid function.

### Backward Propagation

Gradients computed using chain rule:

dZ2 = A2 - Y
dW2 = (1/m) · dZ2 · A1^T
db2 = (1/m) · Σ dZ2

dZ1 = W2^T · dZ2 ⊙ A1 ⊙ (1 - A1)
dW1 = (1/m) · dZ1 · X^T
db1 = (1/m) · Σ dZ1

text

### Parameter Update

W = W - α · dW
b = b - α · db

text

Where α is the learning rate.

## Tips for Better Performance

1. **Normalize inputs**: Data should be in [0, 1] or standardized
2. **Try different architectures**: Experiment with `n_hidden`
3. **Learning rate tuning**: Start with 0.1, adjust if unstable
4. **Early stopping**: Monitor validation loss to prevent overfitting
5. **Weight initialization**: Use Xavier/He initialization for deeper networks

## Limitations

- Only supports binary classification
- Single hidden layer (shallow network)
- No regularization (L1/L2)
- No batch processing or mini-batches
- No advanced optimizers (Adam, RMSprop)

## Extending the Code

### Add More Layers

def initialize_parameters_deep(layer_dims):
parameters = {}
for l in range(1, len(layer_dims)):
parameters[f'W{l}'] = np.random.randn(
layer_dims[l], layer_dims[l-1]
) * 0.01
parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
return parameters

text

### Add Regularization

L2 regularization
lambda_reg = 0.01
loss_reg = loss + (lambda_reg / (2*m)) * (np.sum(W12) + np.sum(W22))
