# Neural Network — Binary Classification from Scratch

A multi-layer neural network implemented using NumPy only. No deep learning framework is used. Forward propagation, backpropagation, and gradient descent are derived and coded explicitly. Applied to binary digit classification on MNIST (digits 0 vs. 1).

## Architecture

```
Input (784) → Dense (64, sigmoid) → Dense (16, sigmoid) → Output (1, sigmoid)
```

Training minimizes binary cross-entropy via full-batch gradient descent.

## Mathematics

### Forward Propagation

For each layer l:

```
Zˡ = Wˡ Aˡ⁻¹ + bˡ
Aˡ = σ(Zˡ)
```

where the sigmoid activation is:

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

### Loss

Binary cross-entropy:

```
L = −(1/m) Σ [ y log(ŷ) + (1−y) log(1−ŷ) ]
```

### Backpropagation

Gradients are computed recursively from the output:

```
δᴸ = Aᴸ − y
δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ σ'(Zˡ)

∂L/∂Wˡ = (1/m) δˡ (Aˡ⁻¹)ᵀ
∂L/∂bˡ = (1/m) Σ δˡ
```

Weights are initialized with the Xavier scheme: W ~ N(0, 1/nˡ⁻¹).

## Results

Trained for 1000 iterations on ~13,000 MNIST samples (digits 0 and 1):

| Split | Accuracy |
|-------|----------|
| Train | ~99.5% |
| Test | ~99.3% |

## Requirements

```
numpy
matplotlib
scikit-learn
tqdm
```

Install:

```bash
pip install numpy matplotlib scikit-learn tqdm
```

## Usage

```bash
python Neural_network.py
```

MNIST is downloaded automatically via `sklearn.datasets.fetch_openml`. The script trains the network and plots loss and accuracy curves.

## Files

| File | Description |
|------|-------------|
| `Neural_network.py` | Full implementation with training pipeline |
| `binary-classification.py` | Standalone binary classification example |

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- LeCun, Y. et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
