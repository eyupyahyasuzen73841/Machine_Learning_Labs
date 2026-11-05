"""
Autograd Engine Verification Suite: Eyüp Yahya Süzen (73841)
Verifies: ((a*b + c)**2).relu().sum() against PyTorch Autograd and Central Differences.
"""

import json
import os
import torch
from core_scalar import GradScalar


def compute_expression(a_vec, b_vec, c_vec):
    items = []
    for a, b, c in zip(a_vec, b_vec, c_vec):
        term = ((a * b + c) ** 2).relu()
        items.append(term)
    
    res = items[0]
    for it in items[1:]:
        res = res + it
    return res


def compute_fd(a_vals, b_vals, c_vals, param_name, idx, h=1e-7):
    def eval_f(a, b, c):
        return sum(max(0.0, (ai * bi + ci) ** 2) for ai, bi, ci in zip(a, b, c))

    a_p, a_m = list(a_vals), list(a_vals)
    b_p, b_m = list(b_vals), list(b_vals)
    c_p, c_m = list(c_vals), list(c_vals)

    if param_name == 'a':
        a_p[idx] += h
        a_m[idx] -= h
    elif param_name == 'b':
        b_p[idx] += h
        b_m[idx] -= h
    else:
        c_p[idx] += h
        c_m[idx] -= h

    return (eval_f(a_p, b_p, c_p) - eval_f(a_m, b_m, c_m)) / (2.0 * h)


def run_checks():
    print("[EYUP SUZEN - GRAD CHECK] Running verification on target formula...")
    a_raw = [1.80, -2.40, 0.90, -1.10]
    b_raw = [-2.50, 1.75, 3.10, 0.80]
    c_raw = [1.10, -0.60, -2.20, 1.40]

    # Custom GradScalar Pass
    a_nodes = [GradScalar(v, name=f'a_{i}') for i, v in enumerate(a_raw)]
    b_nodes = [GradScalar(v, name=f'b_{i}') for i, v in enumerate(b_raw)]
    c_nodes = [GradScalar(v, name=f'c_{i}') for i, v in enumerate(c_raw)]

    loss_custom = compute_expression(a_nodes, b_nodes, c_nodes)
    loss_custom.backward()

    # PyTorch Reference Pass
    ta = torch.tensor(a_raw, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_raw, dtype=torch.float64, requires_grad=True)
    tc = torch.tensor(c_raw, dtype=torch.float64, requires_grad=True)

    loss_torch = (((ta * tb + tc) ** 2).relu()).sum()
    loss_torch.backward()

    # Numerical Finite Difference Pass
    max_torch_diff = 0.0
    max_fd_diff = 0.0
    audit_log = []

    for name, nodes, t_tensor in [('a', a_nodes, ta), ('b', b_nodes, tb), ('c', c_nodes, tc)]:
        for i in range(len(a_raw)):
            g_cust = nodes[i].grad
            g_torch = t_tensor.grad[i].item()
            g_fd = compute_fd(a_raw, b_raw, c_raw, name, i)

            diff_t = abs(g_cust - g_torch)
            diff_f = abs(g_cust - g_fd)

            max_torch_diff = max(max_torch_diff, diff_t)
            max_fd_diff = max(max_fd_diff, diff_f)

            audit_log.append({
                'variable': f"{name}[{i}]",
                'custom_grad': g_cust,
                'torch_grad': g_torch,
                'numerical_grad': g_fd,
                'diff_torch': diff_t,
                'diff_fd': diff_f
            })

    passed = bool(max_torch_diff < 1e-6)
    print(f"  Max Diff vs PyTorch: {max_torch_diff:.2e}")
    print(f"  Max Diff vs Finite Diff: {max_fd_diff:.2e}")
    print(f"  Status: {'PASSED (Threshold < 1e-6)' if passed else 'FAILED'}")

    report_data = {
        'student': 'Eyüp Yahya Süzen (73841)',
        'expression': '((a*b + c)**2).relu().sum()',
        'forward_loss_custom': loss_custom.value,
        'forward_loss_torch': loss_torch.item(),
        'max_torch_error': max_torch_diff,
        'max_numerical_error': max_fd_diff,
        'threshold': 1e-6,
        'verification_passed': passed,
        'details': audit_log
    }

    os.makedirs('results', exist_ok=True)
    with open('results/grad_check_results.json', 'w') as f:
        json.dump(report_data, f, indent=2)

    return report_data


if __name__ == '__main__':
    run_checks()
