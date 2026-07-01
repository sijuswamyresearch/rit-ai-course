"""
Exercise 3.1: Neural Network from Scratch
Objective: Build a neural network without high-level frameworks.

Requirements:
- Implement forward and backward propagation
- Support multiple activation functions (ReLU, Sigmoid, Tanh)
- Train on a simple dataset (e.g., XOR, circles)
- Visualize decision boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# ============================================================
# Part 1: Activation Functions
# ============================================================

class ActivationFunctions:
    """
    Collection of activation functions and their derivatives.
    
    Activation functions introduce non-linearity into the network,
    allowing it to learn complex patterns.
    """
    
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.
        Maps input to range (0, 1).
        
        f(x) = 1 / (1 + e^(-x))
        """
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative of sigmoid function.
        f'(x) = f(x) * (1 - f(x))
        """
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """
        Hyperbolic tangent activation function.
        Maps input to range (-1, 1).
        
        f(x) = tanh(x)
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """
        Derivative of tanh function.
        f'(x) = 1 - tanh^2(x)
        """
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x):
        """
        Rectified Linear Unit (ReLU) activation function.
        f(x) = max(0, x)
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """
        Derivative of ReLU function.
        f'(x) = 1 if x > 0, else 0
        """
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU activation function.
        Allows small gradient when x < 0.
        """
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """
        Derivative of Leaky ReLU function.
        """
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x):
        """
        Softmax activation function for multi-class classification.
        f(x_i) = e^(x_i) / sum(e^(x_j))
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ============================================================
# Part 2: Loss Functions
# ============================================================

class LossFunctions:
    """
    Collection of loss functions for training.
    """
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """
        Binary cross-entropy loss for binary classification.
        L = -[y*log(y_pred) + (1-y)*log(1-y_pred)]
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        """
        Derivative of binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """
        Categorical cross-entropy loss for multi-class classification.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Mean squared error loss for regression.
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        """
        Derivative of MSE loss.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]


# ============================================================
# Part 3: Neural Network Layer
# ============================================================

class Layer:
    """
    A single layer in the neural network.
    
    Each layer has:
    - Weights (W): Learnable parameters
    - Biases (b): Learnable parameters
    - Cache: Stores values for backpropagation
    """
    
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Initialize layer with He initialization for weights.
        
        He initialization is particularly good for ReLU activations
        as it helps prevent vanishing/exploding gradients.
        """
        # He initialization
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((1, output_size))
        
        # Store activation function
        self.activation = activation
        
        # Cache for backpropagation
        self.cache = {}
        
        # Gradients
        self.dW = None
        self.db = None
        
        # Momentum for optimization
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
    
    def forward(self, X):
        """
        Forward pass through the layer.
        
        z = X @ W + b
        a = activation(z)
        """
        self.cache['input'] = X
        
        # Linear transformation
        z = np.dot(X, self.W) + self.b
        self.cache['z'] = z
        
        # Apply activation function
        if self.activation == 'sigmoid':
            a = ActivationFunctions.sigmoid(z)
        elif self.activation == 'tanh':
            a = ActivationFunctions.tanh(z)
        elif self.activation == 'relu':
            a = ActivationFunctions.relu(z)
        elif self.activation == 'leaky_relu':
            a = ActivationFunctions.leaky_relu(z)
        elif self.activation == 'softmax':
            a = ActivationFunctions.softmax(z)
        elif self.activation == 'linear':
            a = z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        self.cache['output'] = a
        return a
    
    def backward(self, da):
        """
        Backward pass through the layer.
        
        Computes gradients with respect to:
        - Weights (dW)
        - Biases (db)
        - Input (da_prev)
        """
        z = self.cache['z']
        a = self.cache['output']
        X = self.cache['input']
        
        # Compute dz based on activation function
        if self.activation == 'sigmoid':
            dz = da * ActivationFunctions.sigmoid_derivative(z)
        elif self.activation == 'tanh':
            dz = da * ActivationFunctions.tanh_derivative(z)
        elif self.activation == 'relu':
            dz = da * ActivationFunctions.relu_derivative(z)
        elif self.activation == 'leaky_relu':
            dz = da * ActivationFunctions.leaky_relu_derivative(z)
        elif self.activation == 'softmax':
            dz = da  # Softmax derivative is handled in loss
        elif self.activation == 'linear':
            dz = da
        
        # Compute gradients
        m = X.shape[0]
        self.dW = np.dot(X.T, dz) / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        
        # Gradient for previous layer
        da_prev = np.dot(dz, self.W.T)
        
        return da_prev
    
    def update(self, learning_rate, optimizer='sgd', momentum=0.9):
        """
        Update weights and biases using gradient descent.
        
        Supports:
        - SGD: Simple gradient descent
        - Momentum: Gradient descent with momentum
        """
        if optimizer == 'sgd':
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db
        elif optimizer == 'momentum':
            self.vW = momentum * self.vW - learning_rate * self.dW
            self.vb = momentum * self.vb - learning_rate * self.db
            self.W += self.vW
            self.b += self.vb


# ============================================================
# Part 4: Neural Network
# ============================================================

class NeuralNetwork:
    """
    A feedforward neural network built from scratch.
    
    Supports:
    - Multiple hidden layers
    - Various activation functions
    - Binary and multi-class classification
    - Regularization (L2)
    """
    
    def __init__(self, layer_sizes, activations, learning_rate=0.01, 
                 l2_lambda=0.0, optimizer='sgd', momentum=0.9):
        """
        Initialize neural network.
        
        Parameters:
        -----------
        layer_sizes : list
            List of layer sizes [input_size, hidden1_size, ..., output_size]
        activations : list
            List of activation functions for each layer (except input)
        learning_rate : float
            Learning rate for gradient descent
        l2_lambda : float
            L2 regularization strength
        optimizer : str
            Optimization algorithm ('sgd' or 'momentum')
        momentum : float
            Momentum coefficient for momentum optimizer
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.optimizer = optimizer
        self.momentum = momentum
        
        # Create layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
    
    def forward(self, X):
        """
        Forward pass through the entire network.
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def backward(self, y_true, y_pred):
        """
        Backward pass through the entire network.
        
        Computes gradients starting from the output layer.
        """
        # Compute output gradient based on loss function
        if self.activations[-1] == 'sigmoid':
            # Binary cross-entropy with sigmoid
            da = (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15)
        elif self.activations[-1] == 'softmax':
            # Categorical cross-entropy with softmax
            da = y_pred - y_true
        else:
            # MSE with linear output
            da = 2 * (y_pred - y_true) / y_true.shape[0]
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            da = layer.backward(da)
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute loss with L2 regularization.
        """
        if self.activations[-1] == 'sigmoid':
            loss = LossFunctions.binary_cross_entropy(y_true, y_pred)
        elif self.activations[-1] == 'softmax':
            loss = LossFunctions.categorical_cross_entropy(y_true, y_pred)
        else:
            loss = LossFunctions.mean_squared_error(y_true, y_pred)
        
        # Add L2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0
            for layer in self.layers:
                l2_loss += np.sum(layer.W ** 2)
            loss += (self.l2_lambda / 2) * l2_loss
        
        return loss
    
    def compute_accuracy(self, y_true, y_pred, task='classification'):
        """
        Compute accuracy for classification tasks.
        """
        if task == 'classification':
            if y_pred.shape[1] > 1:  # Multi-class
                predictions = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y_true, axis=1)
            else:  # Binary
                predictions = (y_pred > 0.5).astype(int).flatten()
                true_labels = y_true.flatten()
            
            return np.mean(predictions == true_labels)
        else:
            # For regression, use R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def train(self, X, y, epochs=1000, batch_size=None, verbose=True, 
              X_val=None, y_val=None, early_stopping_patience=None):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples, n_classes) or (n_samples, 1)
            Training labels
        epochs : int
            Number of training epochs
        batch_size : int or None
            Batch size for mini-batch gradient descent. If None, use full batch.
        verbose : bool
            Whether to print training progress
        X_val, y_val : numpy arrays
            Validation data for early stopping
        early_stopping_patience : int or None
            Number of epochs to wait before early stopping
        """
        n_samples = X.shape[0]
        
        if batch_size is None:
            batch_size = n_samples
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = max(1, n_samples // batch_size)
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss * len(X_batch)
                
                # Backward pass
                self.backward(y_batch, y_pred)
                
                # Update weights
                for layer in self.layers:
                    # Add L2 regularization gradient
                    if self.l2_lambda > 0:
                        layer.dW += self.l2_lambda * layer.W
                    
                    layer.update(self.learning_rate, self.optimizer, self.momentum)
            
            epoch_loss /= n_samples
            self.loss_history.append(epoch_loss)
            
            # Compute training accuracy
            y_pred_train = self.forward(X)
            train_acc = self.compute_accuracy(y, y_pred_train)
            self.accuracy_history.append(train_acc)
            
            # Validation and early stopping
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss = self.compute_loss(y_val, y_pred_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
                
                if verbose and (epoch + 1) % 100 == 0:
                    val_acc = self.compute_accuracy(y_val, y_pred_val)
                    print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - "
                          f"Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {train_acc:.4f}")
        
        return self.loss_history
    
    def predict(self, X):
        """
        Make predictions.
        """
        return self.forward(X)
    
    def predict_classes(self, X):
        """
        Predict class labels.
        """
        y_pred = self.forward(X)
        if y_pred.shape[1] > 1:
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred > 0.5).astype(int).flatten()


# ============================================================
# Part 5: Dataset Generation
# ============================================================

def generate_xor_dataset(n_samples=1000):
    """
    Generate XOR dataset - a classic problem that requires non-linear decision boundary.
    """
    print("Generating XOR dataset...")
    
    # XOR pattern
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ])
    y = np.array([0, 1, 1, 0])
    
    # Add noise to create more samples
    X_noisy = []
    y_noisy = []
    
    samples_per_point = n_samples // 4
    
    for i, (x, label) in enumerate(zip(X, y)):
        for _ in range(samples_per_point):
            noise = np.random.randn(2) * 0.1
            X_noisy.append(x + noise)
            y_noisy.append(label)
    
    X = np.array(X_noisy)
    y = np.array(y_noisy)
    
    return X, y


def generate_circles_dataset(n_samples=1000, noise=0.1):
    """
    Generate concentric circles dataset.
    """
    print("Generating circles dataset...")
    X, y = make_circles(n_samples=n_samples, noise=noise, random_state=42)
    return X, y


def generate_moons_dataset(n_samples=1000, noise=0.1):
    """
    Generate two moons dataset.
    """
    print("Generating moons dataset...")
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y


# ============================================================
# Part 6: Visualization Functions
# ============================================================

def plot_decision_boundary(model, X, y, title="Decision Boundary", 
                          feature_names=None, resolution=0.02):
    """
    Plot decision boundary of a trained model.
    """
    # Set up grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    
    # Make predictions on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_classes(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Decision boundary
    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), 
                   edgecolors='black', s=50)
    axes[0].set_title(title)
    axes[0].set_xlabel(feature_names[0] if feature_names else 'Feature 1')
    axes[0].set_ylabel(feature_names[1] if feature_names else 'Feature 2')
    
    # Training curve
    axes[1].plot(model.loss_history, 'b-', linewidth=2)
    axes[1].set_title('Training Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'decision_boundary_{title.replace(" ", "_").lower()}.png', dpi=150)
    plt.show()


def plot_activation_functions():
    """
    Visualize different activation functions and their derivatives.
    """
    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTIONS VISUALIZATION")
    print("=" * 60)
    
    x = np.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    # Sigmoid
    axes[0, 0].plot(x, ActivationFunctions.sigmoid(x), 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid Activation')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='black', linewidth=0.5)
    
    axes[0, 1].plot(x, ActivationFunctions.sigmoid_derivative(x), 'r-', linewidth=2)
    axes[0, 1].set_title('Sigmoid Derivative')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='black', linewidth=0.5)
    
    # Tanh
    axes[1, 0].plot(x, ActivationFunctions.tanh(x), 'b-', linewidth=2)
    axes[1, 0].set_title('Tanh Activation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='black', linewidth=0.5)
    
    axes[1, 1].plot(x, ActivationFunctions.tanh_derivative(x), 'r-', linewidth=2)
    axes[1, 1].set_title('Tanh Derivative')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='black', linewidth=0.5)
    
    # ReLU
    axes[2, 0].plot(x, ActivationFunctions.relu(x), 'b-', linewidth=2)
    axes[2, 0].set_title('ReLU Activation')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[2, 0].axvline(x=0, color='black', linewidth=0.5)
    
    axes[2, 1].plot(x, ActivationFunctions.relu_derivative(x), 'r-', linewidth=2)
    axes[2, 1].set_title('ReLU Derivative')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[2, 1].axvline(x=0, color='black', linewidth=0.5)
    
    # Leaky ReLU
    axes[3, 0].plot(x, ActivationFunctions.leaky_relu(x), 'b-', linewidth=2)
    axes[3, 0].set_title('Leaky ReLU Activation')
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[3, 0].axvline(x=0, color='black', linewidth=0.5)
    
    axes[3, 1].plot(x, ActivationFunctions.leaky_relu_derivative(x), 'r-', linewidth=2)
    axes[3, 1].set_title('Leaky ReLU Derivative')
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[3, 1].axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150)
    plt.show()


# ============================================================
# Part 7: Experiments
# ============================================================

def experiment_xor():
    """
    Experiment: Solve XOR problem with neural network.
    
    XOR is a classic problem that demonstrates the need for 
    non-linear activation functions and hidden layers.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: XOR PROBLEM")
    print("=" * 60)
    
    # Generate data
    X, y = generate_xor_dataset(n_samples=1000)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert y to column vector
    y_col = y.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_col, test_size=0.2, random_state=42
    )
    
    # Create and train network
    print("\nCreating neural network...")
    nn = NeuralNetwork(
        layer_sizes=[2, 4, 4, 1],
        activations=['relu', 'relu', 'sigmoid'],
        learning_rate=0.1,
        l2_lambda=0.001,
        optimizer='momentum',
        momentum=0.9
    )
    
    print("Training on XOR problem...")
    nn.train(X_train, y_train, epochs=2000, verbose=True,
             X_val=X_test, y_val=y_test, early_stopping_patience=200)
    
    # Evaluate
    y_pred = nn.predict(X_test)
    train_pred = nn.predict(X_train)
    
    train_acc = nn.compute_accuracy(y_train, train_pred)
    test_acc = nn.compute_accuracy(y_test, y_pred)
    
    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Visualize
    plot_decision_boundary(nn, X_scaled, y, title="XOR Decision Boundary",
                          feature_names=['Input 1', 'Input 2'])
    
    return nn


def experiment_circles():
    """
    Experiment: Solve concentric circles problem.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: CONCENTRIC CIRCLES")
    print("=" * 60)
    
    # Generate data
    X, y = generate_circles_dataset(n_samples=1000, noise=0.1)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_col = y.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_col, test_size=0.2, random_state=42
    )
    
    # Create and train network
    print("\nCreating neural network...")
    nn = NeuralNetwork(
        layer_sizes=[2, 8, 8, 4, 1],
        activations=['relu', 'relu', 'relu', 'sigmoid'],
        learning_rate=0.05,
        l2_lambda=0.001,
        optimizer='momentum'
    )
    
    print("Training on circles problem...")
    nn.train(X_train, y_train, epochs=3000, verbose=True,
             X_val=X_test, y_val=y_test, early_stopping_patience=300)
    
    # Evaluate
    y_pred = nn.predict(X_test)
    test_acc = nn.compute_accuracy(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Visualize
    plot_decision_boundary(nn, X_scaled, y, title="Circles Decision Boundary",
                          feature_names=['Feature 1', 'Feature 2'])
    
    return nn


def experiment_moons():
    """
    Experiment: Solve two moons problem.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: TWO MOONS")
    print("=" * 60)
    
    # Generate data
    X, y = generate_moons_dataset(n_samples=1000, noise=0.15)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_col = y.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_col, test_size=0.2, random_state=42
    )
    
    # Create and train network
    print("\nCreating neural network...")
    nn = NeuralNetwork(
        layer_sizes=[2, 16, 16, 8, 1],
        activations=['relu', 'relu', 'relu', 'sigmoid'],
        learning_rate=0.05,
        l2_lambda=0.001,
        optimizer='momentum'
    )
    
    print("Training on moons problem...")
    nn.train(X_train, y_train, epochs=3000, verbose=True,
             X_val=X_test, y_val=y_test, early_stopping_patience=300)
    
    # Evaluate
    y_pred = nn.predict(X_test)
    test_acc = nn.compute_accuracy(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Visualize
    plot_decision_boundary(nn, X_scaled, y, title="Moons Decision Boundary",
                          feature_names=['Feature 1', 'Feature 2'])
    
    return nn


def experiment_activation_comparison():
    """
    Compare different activation functions on the same problem.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: ACTIVATION FUNCTION COMPARISON")
    print("=" * 60)
    
    # Generate data
    X, y = generate_circles_dataset(n_samples=500, noise=0.1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_col = y.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_col, test_size=0.2, random_state=42
    )
    
    # Test different activation functions
    activations_to_test = [
        (['sigmoid', 'sigmoid', 'sigmoid'], "All Sigmoid"),
        (['tanh', 'tanh', 'tanh'], "All Tanh"),
        (['relu', 'relu', 'sigmoid'], "ReLU hidden + Sigmoid output"),
        (['leaky_relu', 'leaky_relu', 'sigmoid'], "Leaky ReLU hidden + Sigmoid output")
    ]
    
    results = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (activations, name) in enumerate(activations_to_test):
        print(f"\nTesting: {name}")
        
        nn = NeuralNetwork(
            layer_sizes=[2, 8, 8, 1],
            activations=activations,
            learning_rate=0.1,
            optimizer='sgd'
        )
        
        nn.train(X_train, y_train, epochs=2000, verbose=False)
        
        y_pred = nn.predict(X_test)
        test_acc = nn.compute_accuracy(y_test, y_pred)
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        results.append({
            'name': name,
            'accuracy': test_acc,
            'loss_history': nn.loss_history
        })
        
        # Plot decision boundary
        axes[idx].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, 
                         cmap=ListedColormap(['#FF0000', '#0000FF']), 
                         alpha=0.5, s=30)
        axes[idx].set_title(f"{name}\nAccuracy: {test_acc:.4f}")
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=150)
    plt.show()
    
    # Plot loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in results:
        ax.plot(result['loss_history'], label=result['name'], linewidth=2)
    ax.set_title('Training Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=150)
    plt.show()
    
    return results


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Exercise 3.1: Neural Network from Scratch")
    print("=" * 60)
    
    # Visualize activation functions
    plot_activation_functions()
    
    # Run experiments
    nn_xor = experiment_xor()
    nn_circles = experiment_circles()
    nn_moons = experiment_moons()
    activation_results = experiment_activation_comparison()
    
    print("\n" + "=" * 60)
    print("EXERCISE COMPLETE!")
    print("=" * 60)
    
    print("\nKey Learnings:")
    print("1. Neural network implementation from scratch")
    print("2. Forward and backward propagation")
    print("3. Different activation functions and their properties")
    print("4. Loss functions for classification")
    print("5. Decision boundary visualization")
    print("6. Impact of network architecture on performance")
    
    print("\nNext Steps:")
    print("- Implement batch normalization")
    print("- Add dropout for regularization")
    print("- Implement Adam optimizer")
    print("- Try multi-class classification")
    print("- Build convolutional neural networks")