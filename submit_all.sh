#!/bin/bash
# ============================================================
# Submit all 9 experiments to Slurm
# Run this from the login node: bash submit_all.sh
# ============================================================

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/sgnanaku
AREAL=$SCRATCH/AReaL
LOGS=$SCRATCH/slurm_logs
mkdir -p $LOGS

# Function to submit one job
submit_job() {
    local EXP_NAME=$1
    local ALGO_FLAGS=$2
    local ETA=$3

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:2
#SBATCH --time=1:00:00
#SBATCH --output=${LOGS}/${EXP_NAME}_%j.out
#SBATCH --error=${LOGS}/${EXP_NAME}_%j.err

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/sgnanaku
MODEL_PATH=\$SCRATCH/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

module load apptainer

apptainer exec --nv --writable-tmpfs \\
  --bind \$SCRATCH:\$SCRATCH \\
  --env "LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib" \\
  --env "HF_DATASETS_OFFLINE=1" \\
  --env "HF_HUB_OFFLINE=1" \\
  --env "HF_HOME=\$SCRATCH/.cache/huggingface" \\
  --env "PYTHONSTARTUP=/tmp/te_stub.py" \\
  \$SCRATCH/areal.sif /bin/bash -c "
    echo \"import sys; sys.modules['transformer_engine'] = type(sys)('transformer_engine')\" > /tmp/te_stub.py
    export LD_LIBRARY_PATH=/.singularity.d/libs:\\\$LD_LIBRARY_PATH
    source /AReaL/.venv/bin/activate
    cd \$SCRATCH/AReaL
    uv pip install -e . --no-deps -q

    python3 examples/math/gsm8k_rl.py \\\\
        --config examples/math/gsm8k_grpo.yaml \\\\
        scheduler.type=local \\\\
        experiment_name=${EXP_NAME} \\\\
        trial_name=run1 \\\\
        rollout.backend=sglang:d1p1t1 \\\\
        actor.backend=fsdp:d1p1t1 \\\\
        cluster.n_nodes=1 \\\\
        cluster.n_gpus_per_node=2 \\\\
        actor.path=\$MODEL_PATH \\\\
        gconfig.max_new_tokens=512 \\\\
        train_dataset.batch_size=64 \\\\
        total_train_epochs=1 \\\\
        rollout.max_head_offpolicyness=${ETA} \\\\
        ${ALGO_FLAGS}
  "
EOF
    echo "Submitted: ${EXP_NAME} (eta=${ETA})"
}

# GRPO (baseline)
submit_job "grpo_eta0" "" "0"
submit_job "grpo_eta2" "" "2"
submit_job "grpo_eta4" "" "4"

# CISPO
submit_job "cispo_eta0" "+actor.use_cispo_loss=true +actor.cispo_epsilon_high=1.2" "0"
submit_job "cispo_eta2" "+actor.use_cispo_loss=true +actor.cispo_epsilon_high=1.2" "2"
submit_job "cispo_eta4" "+actor.use_cispo_loss=true +actor.cispo_epsilon_high=1.2" "4"

# SAPO
submit_job "sapo_eta0" "+actor.use_sapo_loss=true +actor.use_decoupled_loss=false" "0"
submit_job "sapo_eta2" "+actor.use_sapo_loss=true +actor.use_decoupled_loss=false" "2"
submit_job "sapo_eta4" "+actor.use_sapo_loss=true +actor.use_decoupled_loss=false" "4"

echo ""
echo "All 9 jobs submitted! Monitor with: squeue -u sgnanaku"
echo "Logs at: $LOGS"
