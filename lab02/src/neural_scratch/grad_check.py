import numpy as np

def run_gradient_check(model, X_mini, y_mini, h=1e-5):
    """Audits backpropagation gradient precision against two-sided finite difference."""
    model.forward(X_mini, y_mini)
    model.backward()
    
    discrepancies = []
    # Test output layer weights
    w = model.output_layer.W
    ana_grad = model.output_layer.dW
    indices = np.random.RandomState(42).choice(w.size, size=min(8, w.size), replace=False)
    
    flat_w = w.reshape(-1)
    flat_ana = ana_grad.reshape(-1)
    
    for idx in indices:
        orig = flat_w[idx]
        flat_w[idx] = orig + h
        l_high, _ = model.forward(X_mini, y_mini)
        flat_w[idx] = orig - h
        l_low, _ = model.forward(X_mini, y_mini)
        flat_w[idx] = orig
        num_grad = (l_high - l_low) / (2.0 * h)
        
        diff = abs(flat_ana[idx] - num_grad)
        denom = max(abs(flat_ana[idx]), abs(num_grad), 1e-8)
        discrepancies.append(diff / denom)
        
    return float(np.max(discrepancies))
