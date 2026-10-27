import numpy as np

class LinearLayer:
    def __init__(self, in_dim, out_dim, seed=42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dOut):
        N = self.X.shape[0]
        self.grad_W = np.dot(self.X.T, dOut) / N
        self.grad_b = np.sum(dOut, axis=0, keepdims=True) / N
        return np.dot(dOut, self.W.T)

    def apply_gradients(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, Z):
        self.mask = (Z > 0)
        return np.maximum(0, Z)

    def backward(self, dOut):
        return dOut * self.mask

class ScratchNeuralNet:
    def __init__(self, in_features, hidden_features, num_classes, seed=42):
        self.fc1 = LinearLayer(in_features, hidden_features, seed=seed)
        self.act1 = ReLU()
        self.fc2 = LinearLayer(hidden_features, hidden_features, seed=seed+1)
        self.act2 = ReLU()
        self.fc3 = LinearLayer(hidden_features, num_classes, seed=seed+2)

    def forward(self, X, Y=None):
        out1 = self.fc1.forward(X)
        h1 = self.act1.forward(out1)
        out2 = self.fc2.forward(h1)
        h2 = self.act2.forward(out2)
        logits = self.fc3.forward(h2)

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        loss = None
        if Y is not None:
            eps = 1e-12
            loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

        self.probs = probs
        self.Y = Y
        return logits, probs, loss

    def backward(self):
        dZ3 = self.probs - self.Y
        dH2 = self.fc3.backward(dZ3)
        dZ2 = self.act2.backward(dH2)
        dH1 = self.fc2.backward(dZ2)
        dZ1 = self.act1.backward(dH1)
        dX = self.fc1.backward(dZ1)
        return dX

    def step(self, lr):
        self.fc1.apply_gradients(lr)
        self.fc2.apply_gradients(lr)
        self.fc3.apply_gradients(lr)
