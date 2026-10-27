import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_suite.mlp import EyupMLPClassifier
from model_suite.cnn import EyupLeNetVariant
from model_suite.attention import EyupVisionTransformer
from model_suite.profiling_tools import profile_architecture

def run():
    print("[INFO] Initializing Eyup Suzen Vision Architectures...")
    models = {
        "MLP (300-100-10)": EyupMLPClassifier(),
        "LeNet-5 Variant": EyupLeNetVariant(),
        "ViT (49 Patches)": EyupVisionTransformer()
    }

    results = {}
    for name, net in models.items():
        res = profile_architecture(net)
        results[name] = res
        print(f"[{name}] Params: {res['parameters']:,} | MACs: {res['macs']:,} | Latency: {res['latency_ms']:.3f} ms")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmarks.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Benchmark results saved to {out_dir}")

if __name__ == "__main__":
    run()
