import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
    


# Generate dataset


# Convert to PyTorch tensors
## Génération de données d'entraînement (Lorenz) et de test pour le modèle NODE

def lorenz_deriv(state, t, sigma=10.0, beta=8.0/3.0, rho=28.0):
    """Compute the time-derivative of a Lorenz system.

    state: tensor of shape (3,) or (N, 3)
    """
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return torch.stack([dxdt, dydt, dzdt], dim=-1)

def rk4_step(func, state, t, dt, *args):
    """Perform a single RK4 step."""
    k1 = func(state, t, *args)
    k2 = func(state + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = func(state + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = func(state + dt * k3, t + dt, *args)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def integrate_lorenz(initial_state, t, dt, sigma=10.0, beta=8.0/3.0, rho=28.0):
    """Integrate the Lorenz system using RK4.

    initial_state: tensor of shape (3,) or (N, 3)
    Returns: tensor of shape (num_steps+1, 3) or (num_steps+1, N, 3)
    """
    initial_state = torch.as_tensor(initial_state, dtype=torch.float32)
    num_steps = int(t / dt)
    states = torch.empty((num_steps + 1, *initial_state.shape),
                         dtype=initial_state.dtype, device=initial_state.device)
    states[0] = initial_state

    for i in range(num_steps):
        states[i + 1] = rk4_step(lorenz_deriv, states[i], i * dt, dt, sigma, beta, rho)

    return states
print("Lorenz system integration using RK4 method.")
initial_state = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
states = integrate_lorenz(initial_state=initial_state, t=15.0, dt=0.01)
states_test = integrate_lorenz(initial_state=states[-1] + torch.randn(3) * 0.1, t=200.0, dt=0.01)

## DEFINITION DES DONNÉES D'ENTRAÎNEMENT ET DE TEST POUR LE MODÈLE NODE (NORMALISATION DES DONNÉES)

X_train = states[:-1]
y_train = states[1:]      # décalé d'un pas — la vraie cible dynamique

X_test = states_test[:-1]
y_test = states_test[1:]  # même chose pour le test

mu_train, sigma_train = X_train.mean(dim=0), X_train.std(dim=0)
mu_train_y, sigma_train_y = y_train.mean(dim=0), y_train.std(dim=0)

mu_test, sigma_test = X_test.mean(dim=0), X_test.std(dim=0)
mu_test_y, sigma_test_y = y_test.mean(dim=0), y_test.std(dim=0)

X_train = (X_train - mu_train) / sigma_train
y_train = (y_train - mu_train_y) / sigma_train_y

X_test = (X_test - mu_train) / sigma_train
y_test = (y_test - mu_train_y) / sigma_train_y     

class NODEMLP:                              
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = (torch.randn(input_size, hidden_size) * 0.1).requires_grad_()
        self.b1 = (torch.randn(1, hidden_size) * 0.1).requires_grad_()
        self.W2 = (torch.randn(hidden_size, output_size) * 0.1).requires_grad_()
        self.b2 = (torch.randn(1, output_size) * 0.1).requires_grad_()

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = torch.sigmoid(self.z1)  # Hidden layer activation
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        
        return self.z2 # No activation function for output layer because the output is a freely valued function (dh/dt)   
    
    def func(self, state, t):
        return self.forward(state)
    
    def rk4_step(self, func, state, t, dt, *args):
        """Perform a single RK4 step.
        func: function to compute the derivative (dh/dt) caution to func's arguments args and t and do not flip them by the forward's output.
        state: current state (h)
        t: current time
        dt: time step
        args: additional arguments to pass to func"""
        k1 = func(state, t, *args)
        k2 = func(state + 0.5 * dt * k1, t + 0.5 * dt, *args)
        k3 = func(state + 0.5 * dt * k2, t + 0.5 * dt, *args)
        k4 = func(state + dt * k3, t + dt, *args)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def compute_loss(self,state, target):
        """Compute mean squared error loss."""
        h_pred = self.rk4_step(self.func, state, 0, 0.01)  
        return torch.mean((h_pred - target) ** 2)

    def gradients(self, loss):
        loss.backward()
        return {
            'W1': self.W1.grad,
            'b1': self.b1.grad,
            'W2': self.W2.grad,
            'b2': self.b2.grad,
        }
    
    def update_parameters(self, gradients, learning_rate):
        """Update model parameters using the computed gradients."""
        with torch.no_grad():
            self.W1 -= learning_rate * gradients['W1']
            self.b1 -= learning_rate * gradients['b1']
            self.W2 -= learning_rate * gradients['W2']
            self.b2 -= learning_rate * gradients['b2']
        
        self.W1.grad = None
        self.b1.grad = None
        self.W2.grad = None
        self.b2.grad = None

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.01):
        optimizer = torch.optim.Adam([self.W1, self.b1, self.W2, self.b2], lr=learning_rate)
        loss0 = []
        for epoch in tqdm(range(epochs)):
            total_loss = 0.0
            
            optimizer.zero_grad()
            loss = self.compute_loss(X_train, y_train)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()    
            loss0.append(loss.item())
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss / len(X_train)}')
                print(f'Learning rate: {learning_rate}')
            elif epoch >= 2000:
                learning_rate = 0.001  # Décroissance du taux d'apprentissage après 2000 epochs
                
        print(len(loss0))
        plt.plot([k for k in range(epochs)], loss0)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

        # Prédictions finales du modèle (un pas RK4 depuis chaque état de X_train)
        with torch.no_grad():
            y_pred = self.rk4_step(self.func, X_test, 0, 0.01)

        # Un plot par composante : x, y, z
        labels = ['x', 'y', 'z']
        for i, label in enumerate(labels):
            plt.figure()
            plt.plot(y_test[:, i].numpy(), label=f'{label}_data')
            plt.plot(y_pred[:, i].numpy(), '--', label=f'{label}_pred')
            plt.xlabel('Time step')
            plt.ylabel(label)
            plt.title(f'{label}_pred vs {label}_data')
            plt.legend()
            plt.show()


## ===================== EFFECTIVE TRAINING ===================== ##
input_size = 3
hidden_size = 20
output_size = 3
print(X_train[0:100].shape, X_train[0].shape)
model = NODEMLP(input_size, hidden_size, output_size)
model.train(X_train[0], y_train, epochs=15000, learning_rate=0.01)