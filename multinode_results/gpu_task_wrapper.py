#!/usr/bin/env python
import sys
import pickle
import os

# Add parent directory to path to import reproduce_pot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MD_engine import run_step_parallel

if __name__ == "__main__":
    # Load arguments from pickle file
    args_file = sys.argv[1]
    result_file = sys.argv[2]
    local_gpu_id = sys.argv[3]  # GPU ID on this specific node

    # Explicitly set CUDA_VISIBLE_DEVICES to the assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = local_gpu_id
    print(f"[INFO] Wrapper: Set CUDA_VISIBLE_DEVICES={local_gpu_id}")

    with open(args_file, 'rb') as f:
        args = pickle.load(f)

    # Run the simulation with SLURM GPU assignment enabled
    # This tells run_step_parallel to NOT override CUDA_VISIBLE_DEVICES
    try:
        result = run_step_parallel(*args, use_slurm_gpu=True)

        # Save result
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)

        print(f"[SUCCESS] Task completed, result saved to {result_file}")
    except Exception as e:
        print(f"[ERROR] Task failed: {e}")
        import traceback
        traceback.print_exc()
        # Save error result
        with open(result_file, 'wb') as f:
            pickle.dump((0.0, -1.0), f)  # Default error result
