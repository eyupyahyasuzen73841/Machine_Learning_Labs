import numpy as np

class DenseAffineTransform:
    """Fully vectorized Affine layer W^T * X + b with uniform Xavier/He initialization."""
    def __init__(self, in_dim, out_dim, seed=73841):
        rng = np.random.RandomState(seed)
        bound = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-bound, bound, (in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x_cached = None

    def forward(self, x):
        self.x_cached = x
        return np.dot(x, self.W) + self.b

    def backward(self, delta):
        m = self.x_cached.shape[0]
        self.dW = np.dot(self.x_cached.T, delta) / m
        self.db = np.sum(delta, axis=0, keepdims=True) / m
        return np.dot(delta, self.W.T)

    def optimize(self, alpha):
        self.W -= alpha * self.dW
        self.b -= alpha * self.db

class SigmoidActivation:
    """Logistic sigmoid non-linear activation with numerical clipping."""
    def __init__(self):
        self.out = None

    def forward(self, z):
        z_clamped = np.clip(z, -30.0, 30.0)
        self.out = 1.0 / (1.0 + np.exp(-z_clamped))
        return self.out

    def backward(self, delta):
        return delta * self.out * (1.0 - self.out)

class SoftmaxCrossEntropyLoss:
    """Fused Softmax with categorical cross entropy loss."""
    @staticmethod
    def forward(logits, labels):
        m = logits.shape[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.sum(np.log(np.clip(probs[np.arange(m), labels], 1e-15, 1.0))) / m
        return loss, probs

    @staticmethod
    def backward(probs, labels):
        m = probs.shape[0]
        grad = probs.copy()
        grad[np.arange(m), labels] -= 1.0
        return grad
