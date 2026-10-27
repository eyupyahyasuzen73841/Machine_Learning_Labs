import time
import torch
from thop import profile as thop_profile

def profile_architecture(model, input_shape=(1, 1, 28, 28), trials=80):
    """Quantifies model parameter count, FLOPs / MACs, and microsecond-level inference latency."""
    model.eval()
    dummy = torch.randn(*input_shape)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    
    # Warmup
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy)
        
        t0 = time.perf_counter()
        for _ in range(trials):
            _ = model(dummy)
        t_elapsed = time.perf_counter() - t0
        
    latency_ms = (t_elapsed / trials) * 1000.0
    return {
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "macs": int(macs),
        "mflops": (macs * 2) / 1e6,
        "latency_ms": latency_ms
    }
