# !/usr/bin/env python3
# ============================================================
# Extract and display results from all AReaL experiment logs
# Usage: python3 get_results.py [username]
# Example: python3 get_results.py sgnanaku
# ============================================================
 
import re
import glob
import sys
import os
 
username = sys.argv[1] if len(sys.argv) > 1 else "sgnanaku"
log_dir = f"/scratch/zt1/project/zaoxing-prj/user/{username}/slurm_logs"
 
# Map experiment names to their log files (pick the latest one)
experiments = [
    "grpo_eta0", "grpo_eta2", "grpo_eta4",
    "cispo_eta0", "cispo_eta2", "cispo_eta4",
    "sapo_eta0", "sapo_eta2", "sapo_eta4",
]
 
def find_latest_log(exp_name, log_dir):
    pattern = os.path.join(log_dir, f"{exp_name}_*.out")
    files = glob.glob(pattern)
    if not files:
        # also try with _final_ suffix
        pattern = os.path.join(log_dir, f"{exp_name}_final_*.out")
        files = glob.glob(pattern)
    if not files:
        return None
    # Return the one with training data, preferring latest
    files_with_data = []
    for f in files:
        with open(f) as fh:
            content = fh.read()
        if "Train step" in content:
            files_with_data.append(f)
    if files_with_data:
        return sorted(files_with_data)[-1]
    return sorted(files)[-1]
 
print(f"\n{'='*60}")
print(f"AReaL Experiment Results - User: {username}")
print(f"{'='*60}")
print(f"{'Experiment':<15} {'Final Reward':>12} {'Max Reward':>10} {'Steps':>8} {'Status':>10}")
print("-" * 60)
 
results = {}
for exp in experiments:
    log_file = find_latest_log(exp, log_dir)
    if not log_file:
        print(f"{exp:<15} {'N/A':>12} {'N/A':>10} {'N/A':>8} {'NO LOG':>10}")
        continue
 
    with open(log_file) as f:
        content = f.read()
 
    rewards = re.findall(r'task_reward/avg\s*│\s*([\d.e+\-]+)', content)
    steps = re.findall(r'Train step (\d+)/\d+ done', content)
 
    if rewards:
        final_reward = float(rewards[-1])
        max_reward = max(float(r) for r in rewards)
        n_steps = steps[-1] if steps else "?"
        status = "OK"
        results[exp] = final_reward
        print(f"{exp:<15} {final_reward:>12.4f} {max_reward:>10.4f} {n_steps:>8} {status:>10}")
    else:
        print(f"{exp:<15} {'N/A':>12} {'N/A':>10} {'N/A':>8} {'CRASHED':>10}")
 
print(f"\n{'='*60}")
print("SUMMARY TABLE (Final Reward)")
print(f"{'='*60}")
print(f"{'Algorithm':<10} {'eta=0':>10} {'eta=2':>10} {'eta=4':>10}")
print("-" * 45)
for algo in ["grpo", "cispo", "sapo"]:
    r0 = results.get(f"{algo}_eta0", float('nan'))
    r2 = results.get(f"{algo}_eta2", float('nan'))
    r4 = results.get(f"{algo}_eta4", float('nan'))
    print(f"{algo.upper():<10} {r0:>10.4f} {r2:>10.4f} {r4:>10.4f}")
