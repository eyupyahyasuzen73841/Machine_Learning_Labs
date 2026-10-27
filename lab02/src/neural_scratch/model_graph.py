import numpy as np
from .tensor_layers import DenseAffineTransform, SigmoidActivation, SoftmaxCrossEntropyLoss

class EyupMultiLayerNet:
    """Configurable Multi-Layer Perceptron architecture."""
    def __init__(self, layer_dims=[64, 40, 20, 10], seed=73841):
        self.layers = []
        self.activations = []
        for i in range(len(layer_dims) - 2):
            self.layers.append(DenseAffineTransform(layer_dims[i], layer_dims[i+1], seed=seed + i))
            self.activations.append(SigmoidActivation())
        self.output_layer = DenseAffineTransform(layer_dims[-2], layer_dims[-1], seed=seed + 99)
        self.cached_probs = None
        self.cached_labels = None

    def forward(self, x, labels=None):
        cur = x
        for layer, act in zip(self.layers, self.activations):
            cur = act.forward(layer.forward(cur))
        logits = self.output_layer.forward(cur)

        if labels is not None:
            loss, probs = SoftmaxCrossEntropyLoss.forward(logits, labels)
            self.cached_probs = probs
            self.cached_labels = labels
            return loss, probs
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def backward(self):
        d_out = SoftmaxCrossEntropyLoss.backward(self.cached_probs, self.cached_labels)
        cur_grad = self.output_layer.backward(d_out)
        for layer, act in reversed(list(zip(self.layers, self.activations))):
            cur_grad = layer.backward(act.backward(cur_grad))

    def step(self, lr):
        for layer in self.layers:
            layer.optimize(lr)
        self.output_layer.optimize(lr)
