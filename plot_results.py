#!/usr/bin/env python3
# ============================================================
# Plot results from AReaL experiment logs
# Usage: python3 plot_results.py [username]
# Saves plots to: slurm_logs/plots/
# ============================================================

import re
import glob
import sys
import os
import json

username = sys.argv[1] if len(sys.argv) > 1 else "sgnanaku"
log_dir = f"/scratch/zt1/project/zaoxing-prj/user/{username}/slurm_logs"
plot_dir = os.path.join(log_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Installing matplotlib...")
    os.system("pip install matplotlib numpy --quiet")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

# ============================================================
# Config
# ============================================================
COLORS = {
    'grpo': '#2196F3',   # blue
    'cispo': '#FF5722',  # orange-red
    'sapo': '#4CAF50',   # green
}
LINESTYLES = {
    0: '-',
    2: '--',
    4: ':',
}
ETA_LABELS = {0: 'η=0 (sync)', 2: 'η=2', 4: 'η=4'}
ALGO_LABELS = {'grpo': 'GRPO', 'cispo': 'CISPO', 'sapo': 'SAPO'}

EXPERIMENTS = {
    'grpo_eta0':  ('grpo', 0),
    'grpo_eta2':  ('grpo', 2),
    'grpo_eta4':  ('grpo', 4),
    'cispo_eta0': ('cispo', 0),
    'cispo_eta2': ('cispo', 2),
    'cispo_eta4': ('cispo', 4),
    'sapo_eta0':  ('sapo', 0),
    'sapo_eta2':  ('sapo', 2),
    'sapo_eta4':  ('sapo', 4),
}

# ============================================================
# Parse logs
# ============================================================
def find_latest_log(exp_name, log_dir):
    for pattern in [f"{exp_name}_final_*.out", f"{exp_name}_*.out"]:
        files = glob.glob(os.path.join(log_dir, pattern))
        files_with_data = [f for f in files if "Train step" in open(f).read()]
        if files_with_data:
            return sorted(files_with_data)[-1]
    return None

def parse_log(log_file):
    with open(log_file) as f:
        content = f.read()
    rewards = [float(x) for x in re.findall(r'task_reward/avg\s*│\s*([\d.e+\-]+)', content)]
    grad_norms = [float(x) for x in re.findall(r'update/grad_norm\s*│\s*([\d.e+\-]+)', content)]
    entropy = [float(x) for x in re.findall(r'update/entropy/avg\s*│\s*([\d.e+\-]+)', content)]
    return {
        'rewards': rewards,
        'grad_norms': grad_norms,
        'entropy': entropy,
        'steps': list(range(1, len(rewards) + 1)),
    }

data = {}
for exp_name, (algo, eta) in EXPERIMENTS.items():
    log_file = find_latest_log(exp_name, log_dir)
    if log_file:
        data[exp_name] = parse_log(log_file)
        data[exp_name]['algo'] = algo
        data[exp_name]['eta'] = eta
        print(f"Loaded {exp_name}: {len(data[exp_name]['rewards'])} steps")
    else:
        print(f"Missing: {exp_name}")

# ============================================================
# Plot 1: Reward curves per algorithm (3 subplots)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Reward Curves by Algorithm (GRPO vs CISPO vs SAPO)', fontsize=14, fontweight='bold')

for ax, algo in zip(axes, ['grpo', 'cispo', 'sapo']):
    for eta in [0, 2, 4]:
        exp = f"{algo}_eta{eta}"
        if exp in data and data[exp]['rewards']:
            d = data[exp]
            ax.plot(d['steps'], d['rewards'],
                    color=COLORS[algo],
                    linestyle=LINESTYLES[eta],
                    linewidth=2,
                    label=ETA_LABELS[eta])
    ax.set_title(ALGO_LABELS[algo], fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Task Reward' if algo == 'grpo' else '')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

plt.tight_layout()
path1 = os.path.join(plot_dir, 'reward_curves_by_algo.png')
plt.savefig(path1, dpi=150, bbox_inches='tight')
print(f"Saved: {path1}")
plt.close()

# ============================================================
# Plot 2: Reward curves per staleness level (3 subplots)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Reward Curves by Staleness Level (η=0, 2, 4)', fontsize=14, fontweight='bold')

for ax, eta in zip(axes, [0, 2, 4]):
    for algo in ['grpo', 'cispo', 'sapo']:
        exp = f"{algo}_eta{eta}"
        if exp in data and data[exp]['rewards']:
            d = data[exp]
            ax.plot(d['steps'], d['rewards'],
                    color=COLORS[algo],
                    linewidth=2,
                    label=ALGO_LABELS[algo])
    ax.set_title(ETA_LABELS[eta], fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Task Reward' if eta == 0 else '')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

plt.tight_layout()
path2 = os.path.join(plot_dir, 'reward_curves_by_eta.png')
plt.savefig(path2, dpi=150, bbox_inches='tight')
print(f"Saved: {path2}")
plt.close()

# ============================================================
# Plot 3: Final reward heatmap (3x3 grid)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

algos = ['GRPO', 'CISPO', 'SAPO']
etas = ['η=0', 'η=2', 'η=4']
matrix = np.zeros((3, 3))

for i, algo in enumerate(['grpo', 'cispo', 'sapo']):
    for j, eta in enumerate([0, 2, 4]):
        exp = f"{algo}_eta{eta}"
        if exp in data and data[exp]['rewards']:
            matrix[i, j] = data[exp]['rewards'][-1]

im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.7, vmax=0.95)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(etas, fontsize=12)
ax.set_yticklabels(algos, fontsize=12)
ax.set_xlabel('Staleness (η)', fontsize=12)
ax.set_ylabel('Algorithm', fontsize=12)
ax.set_title('Final Reward Heatmap', fontsize=14, fontweight='bold')

for i in range(3):
    for j in range(3):
        val = matrix[i, j]
        ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='black' if 0.75 < val < 0.92 else 'white')

plt.colorbar(im, ax=ax, label='Final Reward')
plt.tight_layout()
path3 = os.path.join(plot_dir, 'final_reward_heatmap.png')
plt.savefig(path3, dpi=150, bbox_inches='tight')
print(f"Saved: {path3}")
plt.close()

# ============================================================
# Plot 4: Staleness effect (line plot, final reward vs eta)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for algo in ['grpo', 'cispo', 'sapo']:
    rewards = []
    valid_etas = []
    for eta in [0, 2, 4]:
        exp = f"{algo}_eta{eta}"
        if exp in data and data[exp]['rewards']:
            rewards.append(data[exp]['rewards'][-1])
            valid_etas.append(eta)
    if rewards:
        ax.plot(valid_etas, rewards,
                color=COLORS[algo],
                marker='o',
                markersize=10,
                linewidth=2.5,
                label=ALGO_LABELS[algo])
        for eta, r in zip(valid_etas, rewards):
            ax.annotate(f'{r:.4f}', (eta, r),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center', fontsize=9)

ax.set_xlabel('Staleness Level (η)', fontsize=12)
ax.set_ylabel('Final Task Reward', fontsize=12)
ax.set_title('Effect of Staleness on Final Reward', fontsize=14, fontweight='bold')
ax.set_xticks([0, 2, 4])
ax.set_xticklabels(['η=0\n(sync)', 'η=2\n(default)', 'η=4\n(high)'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.7, 1.0)

plt.tight_layout()
path4 = os.path.join(plot_dir, 'staleness_effect.png')
plt.savefig(path4, dpi=150, bbox_inches='tight')
print(f"Saved: {path4}")
plt.close()

print(f"\nAll plots saved to: {plot_dir}")
