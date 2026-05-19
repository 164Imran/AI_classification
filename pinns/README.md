# Physics-Informed Neural Networks (PINNs)

Implementation of a **Physics-Informed Neural Network** from scratch in PyTorch to solve the **1D heat equation** (parabolic PDE).

---

## What is a PINN?

A PINN is a neural network trained not only on data, but directly on the physics of a problem — encoded as a PDE. Instead of fitting labeled samples, the network learns a solution `u(x, t)` that satisfies:

- the **governing equation** (the PDE itself)
- the **initial condition** (IC)
- the **boundary conditions** (BC)

All three constraints are embedded directly into the loss function. No numerical solver, no grid, no discretization.

---

## Problem: 1D Heat Equation

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [0,1],\ t \in [0,1]$$

with:

| Constraint | Expression |
|---|---|
| Initial condition | `u(x, 0) = 0` |
| Boundary condition (left) | `u(0, t) = 0` |
| Boundary condition (right) | `u(1, t) = 0` |
| Diffusivity | `α = 0.01` |

The network learns the function `u(x, t)` that satisfies this system over the entire domain.

---

## Architecture

```
Input  →  [x, t]           (2 neurons)
Hidden →  8 neurons, tanh
Output →  u(x, t)          (1 neuron, linear)
```

Notation: `[2, 8, 1]`

Weights are initialized with **LeCun initialization** (`σ = 1/√fan_in`), biases at zero. All tensors use `float64` for numerical precision when computing second-order derivatives.

---

## Loss Function

The total loss combines three terms:

$$\mathcal{L} = \mathcal{L}_{\text{res}} + \mathcal{L}_{\text{IC}} + \mathcal{L}_{\text{BC}}$$

### Physics Residual — `L_res`

Evaluated at **collocation points** sampled uniformly inside the domain:

$$\mathcal{L}_{\text{res}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \left( \frac{\partial u}{\partial t}(x_i, t_i) - \alpha \frac{\partial^2 u}{\partial x^2}(x_i, t_i) \right)^2$$

The partial derivatives are computed with `torch.func.grad` (functional API), enabling exact automatic differentiation through the network.

### Initial Condition — `L_IC`

$$\mathcal{L}_{\text{IC}} = \frac{1}{N_{ic}} \sum_{i=1}^{N_{ic}} \left( u(x_i, 0) - 0 \right)^2$$

### Boundary Conditions — `L_BC`

Dirichlet BCs at `x = 0` and `x = 1`:

$$\mathcal{L}_{\text{BC}} = \frac{1}{N_{bc}} \sum_{i=1}^{N_{bc}} \left( u(x_{bc,i}, t_i) - 0 \right)^2$$

---

## Automatic Differentiation with `torch.func`

The key challenge in PINNs is computing spatial and temporal derivatives of the network output. This implementation uses PyTorch's **functional transform API**:

```python
du_dt  = grad(u, argnums=1)(x, t)        # ∂u/∂t
du_dx  = grad(u, argnums=0)              # ∂u/∂x (function)
du_dxx = grad(du_dx, argnums=0)(x, t)   # ∂²u/∂x²
```

`vmap` is used to efficiently batch these scalar operations across all collocation/IC/BC points without explicit loops:

```python
r = vmap(lambda x, t: residual(params, x, t))(x_col, t_col)
```

---

## Training

| Hyperparameter | Value |
|---|---|
| Collocation points (`N_col`) | 1000 |
| IC points (`N_ic`) | 100 |
| BC points (`N_bc`) | 100 |
| Iterations | 4000 |
| Initial learning rate | 0.05 |
| LR decay (after iter 1200) | 0.01 |
| Optimizer | Gradient descent (manual, via `torch.func.grad`) |

Training is fully functional — no `torch.nn.Module`, no `.backward()`. Gradients of the loss with respect to all parameters are computed via `grad(loss_physics, argnums=0)(params, ...)`.

---

## Sampling Strategy

Points are sampled randomly (Monte Carlo) on each run:

```
x_col, t_col  ~ Uniform([0,1])²         # interior collocation
x_ic          ~ Uniform([0,1]), t=0      # initial condition line
x_bc = {0,1}, t_bc ~ Uniform([0,1])     # left and right boundaries
```

This avoids grid artifacts and generalizes well to irregular domains.

---

## Output

After training, the network is evaluated on a 1000×1000 grid over `[0,10]×[0,10]` and visualized as a 3D surface `u(x, t)`:

```python
ax.plot_surface(X.numpy(), T.numpy(), U.numpy(), cmap='viridis')
```

---

## Plots

<!-- Add your plots here -->

---

## Dependencies

```
torch
numpy
matplotlib
tqdm
scikit-learn
```

```bash
pip install torch numpy matplotlib tqdm scikit-learn
```

---

## Run

```bash
python Neural_network.py
```

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- PyTorch functional API: [`torch.func`](https://pytorch.org/docs/stable/func.html)
