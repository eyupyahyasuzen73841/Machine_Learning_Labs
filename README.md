# Machine Learning Laboratory Coursework — Systems & Implementations

**Student Researcher:** Eyüp Yahya Süzen  
**Student ID:** 73841  
**Curriculum:** Machine Learning Systems (Academic Year 2025/2026)  

---

## Overview

This repository contains the complete, verified, four-part laboratory coursework for Machine Learning Systems. All source code, empirical logs, serialized artifacts, and markdown reports are organized into dedicated laboratory folders:

- **`lab01/` — Baseline System Measurements:** Diagnostic classification benchmarks on the Breast Cancer dataset with training time, inference latency, memory profile, and TinyML hardware feasibility.
- **`lab02/` — Neural Network from Scratch:** Full NumPy implementation of a feedforward neural network, analytical matrix backpropagation formulas, finite-difference gradient checks, and training loss curves.
- **`lab03/` — Architectures and Their Costs:** Comparative study of MLP, CNN, and Vision Transformer models on Fashion-MNIST, comparing theoretical FLOPs against `thop` MACs and Pareto trade-off curves.
- **`lab04/` — Data Pipeline and Quality Engineering:** Production data engineering pipeline for Adult Census data with schema contracts, zero data leakage, and label conflict auditing.

## Instructions for Reproduction
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
