# Machine Learning Systems Laboratory Portfolio

**Student Researcher:** Eyüp Yahya Süzen  
**Student ID:** 73841  
**Institution:** Vistula University  
**Academic Year:** 2025/2026  
**Primary Reference:** *Machine Learning Systems* (Vijay Janapa Reddi, Harvard University)  

---

## Coursework Overview and Unified Portfolio Architecture

This repository organizes the comprehensive laboratory curriculum for Machine Learning Systems, structured across four modular investigative modules following strict reproducible engineering standards:

```
Machine_Learning_Labs/
├── README.md               # Repository roadmap and systems specifications
├── requirements.txt        # Strictly pinned dependency manifest
├── .gitignore              # Production-grade cache and artifact exclusions
├── lab01/                  # Lab 01: Systems Profiling & Environment Setup
│   ├── report.md           # Systems measurement audit & deployment tier classification
│   ├── src/                # Baseline model training and memory profiler harness
│   └── results/            # Profiling CSVs, model binaries, and telemetry
├── lab02/                  # Lab 02: Modular Neural Network from First Principles
│   ├── report.md           # Analytical calculus, finite-difference verification, training curves
│   ├── src/                # Vectorized NumPy tensor network and backpropagation engine
│   └── results/            # Multi-parameter convergence logs, loss plots, timing metrics
├── lab03/                  # Lab 03: Architectural Topologies and Execution Budgets
│   ├── report.md           # MLP vs CNN vs Attention trade-offs, theoretical FLOPs vs MACs
│   ├── src/                # PyTorch architecture declarations and benchmarking scripts
│   └── results/            # Pareto frontier plots, latency traces, parameter tables
└── lab04/                  # Lab 04: Industrial Data Engineering & Quality Pipeline
    ├── report.md           # Zero-leakage ETL validation, feature synthesis, label noise audit
    ├── src/                # Production pipeline orchestrator and transformation modules
    └── results/            # Data card, validation telemetry, serialized joblib artifacts
```

## Setup and Execution

```bash
# 1. Environment initialization
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Dependency installation
pip install -r requirements.txt
```
