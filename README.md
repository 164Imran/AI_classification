# Neural Networks — From Scratch

Implementations of neural networks built without high-level frameworks, with every mathematical operation made explicit: weight initialization, forward propagation, backpropagation, and loss computation.
Each project targets a different problem class — classification, regression, and physics-informed learning — and is self-contained in its own subfolder.

## Projects

### Binary Classification — MLP NumPy
Multi-layer perceptron trained on MNIST (digits 0 vs 1). Manual backpropagation, binary cross-entropy loss, Xavier initialization. Also includes a CNN built from scratch with manual convolution, ReLU, max pooling, and softmax.
**Stack:** NumPy, scikit-learn · [`./classification/`](./classification/)

### Curve Regression — MLP NumPy
MLP mapping a circle point distribution to a star shape, trained with MSE loss and gradient descent implemented from scratch.
**Stack:** NumPy · [`./mse_star/`](./mse_star/)

### PINN — Heat Equation
Physics-Informed Neural Network solving the 1D heat equation from scratch. The PDE residual, initial condition, and boundary conditions are embedded directly into the loss. Partial derivatives are computed via `torch.func.grad` and batched with `vmap` — no `nn.Module`, no `.backward()`.
**Stack:** PyTorch · [`./pinns/`](./pinns/)
