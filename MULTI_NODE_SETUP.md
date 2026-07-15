# Multi-Node GPU Execution Setup

## Overview
The code has been modified to automatically detect and use GPUs across multiple SLURM nodes. This allows you to scale beyond the GPU limit of a single node.

## Changes Made

### 1. Modified `calc_state()` function
- **Automatically detects** if running in a multi-node SLURM environment
- **Falls back** to single-node `ProcessPoolExecutor` if only 1 node is allocated
- No changes needed to your existing code that calls `calc_state()`

### 2. Added `get_slurm_nodes()` function
- Parses `SLURM_NODELIST` or `SLURM_JOB_NODELIST` environment variables
- Uses `scontrol show hostnames` to expand node lists
- Handles formats like `node[001-002]` or `node001,node002`

### 3. Added `run_multinode_slurm()` function
- Distributes GPU tasks across nodes using **round-robin** assignment
- Uses `srun` with `--nodelist` to launch tasks on specific nodes
- Creates temporary pickle files for inter-process communication
- Monitors task completion and collects results

## How to Use

### SLURM Script Modifications

**Update your SLURM script** to request 2 nodes:

```bash
#!/bin/bash
#SBATCH --job-name=multi_node_job
#SBATCH --account=chm250129-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=2                    # ← CHANGE THIS from 1 to 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4            # 4 GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --time=40:30:00
#SBATCH --output=train_out_%j.out
#SBATCH --error=train_err_%j.err

# ... rest of your script ...

python -u main_val.py \
  --epochs 110 --batch 4 --first_run 1 --Nc 10000000 \
  --sigma_hi 0.40 --sigma_lo 0.005 --str_index 5 --three_d 1 \
  --pretrain_target 1.00 0.5 0.5 --success_threshold 0.6
```

### Execution Flow

With 2 nodes and 4 GPUs per node (8 total GPUs):

```
Node 1: GPUs 0, 1, 2, 3 (physical)
Node 2: GPUs 0, 1, 2, 3 (physical)

Task Distribution (batch_size=8):
- Task 0 (gpu_id=0) → Node 1, physical GPU assigned by SLURM
- Task 1 (gpu_id=1) → Node 2, physical GPU assigned by SLURM
- Task 2 (gpu_id=2) → Node 1, physical GPU assigned by SLURM
- Task 3 (gpu_id=3) → Node 2, physical GPU assigned by SLURM
- Task 4 (gpu_id=4) → Node 1, physical GPU assigned by SLURM
- Task 5 (gpu_id=5) → Node 2, physical GPU assigned by SLURM
- Task 6 (gpu_id=6) → Node 1, physical GPU assigned by SLURM
- Task 7 (gpu_id=7) → Node 2, physical GPU assigned by SLURM
```

**Important**: `gpu_id` (0-7) is used only for:
- File naming (e.g., `reproduce_{gpu_id}.gsd`)
- Random seed generation
- Task identification

**Physical GPU assignment** is handled automatically by SLURM via `--gpus-per-task=1`.

## Key Features

### Automatic Detection
- If `SLURM_NODELIST` contains 1 node → uses single-node `ProcessPoolExecutor`
- If `SLURM_NODELIST` contains 2+ nodes → uses multi-node `srun` execution

### No Code Changes Required
- Your existing Python code doesn't need modification
- Just update the SLURM script to request more nodes

### Fault Tolerance
- If a task fails, it returns default values `(0.0, -1.0)`
- Pipeline continues without crashing

### Shared Filesystem
- Uses temporary pickle files in `multinode_results/` directory
- Files are cleaned up after results are collected

## Monitoring

Check progress with:
```bash
# View SLURM output
tail -f train_out_<jobid>.out

# Check GPU usage on nodes
srun --jobid=<jobid> --pty bash
nvidia-smi
```

Look for log messages:
```
[INFO] Multi-node execution detected: 2 nodes
[INFO] Distributing 8 GPU tasks across nodes: ['node001', 'node002']
[INFO] Launching task 0 on node node001 (GPU 0)
[SUCCESS] Task 0 on node001 completed
```

## Troubleshooting

### Issue: Tasks not distributing
**Solution**: Ensure SLURM environment variables are set:
```bash
echo $SLURM_NODELIST
echo $SLURM_JOB_NODELIST
```

### Issue: GPU conflicts
**Solution**: The code uses `--exclusive` flag in srun to prevent conflicts

### Issue: File not found errors
**Solution**: Ensure all nodes can access the shared filesystem where your code is located

### Issue: Still using single-node execution
**Check**:
- Verify `--nodes=2` in SLURM script
- Check `scontrol show job <jobid>` to see allocated nodes
- Look for log message: `[INFO] Multi-node execution detected`

## Testing

### Test with 1 node (should use ProcessPoolExecutor):
```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
```

Output: `[INFO] Single-node execution`

### Test with 2 nodes (should use srun):
```bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
```

Output: `[INFO] Multi-node execution detected: 2 nodes`

## Performance

**Expected speedup**:
- 2 nodes with 4 GPUs each = 8 GPUs total
- With `batch_size=8`, all simulations run in parallel
- ~2x faster than single node (4 GPUs)

## Files Created

The code creates:
- `multinode_results/` - temporary directory for task coordination
- `multinode_results/gpu_task_wrapper.py` - wrapper script for srun execution
- `multinode_results/task_<i>_args.pkl` - temporary argument files (cleaned up)
- `multinode_results/task_<i>_result.pkl` - temporary result files (cleaned up)

## Important Notes

1. **Shared filesystem required**: All nodes must access the same working directory
2. **GPU numbering**:
   - `gpu_id` (0 to batch_size-1) is used for file naming and seeding
   - Physical GPU assignment is handled by SLURM automatically
   - Each task gets exactly 1 GPU via `--gpus-per-task=1`
3. **CUDA_VISIBLE_DEVICES**:
   - **Multi-node**: Set automatically by SLURM (do NOT override)
   - **Single-node**: Manually set to `gpu_id` for backward compatibility
4. **Network latency**: Some overhead from file I/O, but minimal for long-running simulations

## How GPU Assignment Works

### Single-Node Mode (--nodes=1):
- `gpu_id` directly controls `CUDA_VISIBLE_DEVICES`
- Traditional behavior: `gpu_id=0` → GPU 0, `gpu_id=1` → GPU 1, etc.

### Multi-Node Mode (--nodes=2+):
- SLURM assigns physical GPUs via `srun --gpus-per-task=1`
- `CUDA_VISIBLE_DEVICES` is set by SLURM (e.g., to "0" or "2")
- `gpu_id` is used only for file names and random seeds
- Example: Task 4 (gpu_id=4) on Node 1 might get physical GPU 0 or 1 or 2 or 3 (SLURM decides)

## Contact

If you encounter issues, check:
1. SLURM output/error files
2. GPU availability: `squeue -u $USER`
3. Node status: `sinfo -N`
