# Neural Network — MSE Loss on a Star Curve

Neural network implemented from scratch with NumPy, trained with **MSE loss** to learn a geometric transformation: mapping a circle to a star-shaped curve.

---

## Task

The network learns the mapping:

```
X : circle (parametric, 100 points in R²)
y : star   (5-branch, interpolated, 100 points in R²)
```

Both are 2D curves parametrized by `t ∈ [0, 2π]`. The network is trained to predict `y` from `X` purely by minimizing squared error — no labels, no classes, just regression on curve coordinates.

---

## Architecture

```
Input  →  2 neurons  (x, y coordinates of the circle)
Hidden →  32 neurons, tanh activation
Output →  2 neurons  (x, y coordinates of the star)
```

Notation: `[2, 32, 2]`

Weights initialized with LeCun initialization, biases at zero, fixed seed for reproducibility.

---

## Implementation

Everything is implemented from scratch with NumPy:

- **Forward pass**: layer-by-layer matrix multiply + tanh (hidden), linear (output)
- **Backward pass**: manual backpropagation, MSE gradient
- **Update**: vanilla gradient descent

Loss at each step:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \| \hat{y}_i - y_i \|^2$$

---

## Training

| Hyperparameter | Value |
|---|---|
| Iterations | 15 000 |
| Initial learning rate | 0.05 |
| LR decay (after iter 1200) | 0.01 |
| Optimizer | Gradient descent |
| Loss | MSE |

---

## Output

Animated plot with two panels:
- **Left**: predicted curve vs target star, updated at each logged iteration
- **Right**: MSE loss curve over training

---

## Dependencies

```
numpy
matplotlib
tqdm
scikit-learn
```

```bash
pip install numpy matplotlib tqdm scikit-learn
```

---

## Run

```bash
python Neural_network.py
```
