import json
import multiprocessing
import os
import pickle
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

import freud
import gsd
import gsd.hoomd
import hoomd
import hoomd.custom
import hoomd.md as md
import hoomd.variant
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.distributions import Normal, TransformedDistribution, TanhTransform

import ovito
from ovito.io import import_file
from ovito.modifiers import *
from ovito.vis import *

warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import ovito._extensions.pyscript
from ovito.modifiers import ClusterAnalysisModifier

from hoomd import box as hoomd_box
from hoomd.md.methods import Langevin
from hoomd.md.pair import Table
from hoomd.md.nlist import Cell
from hoomd.variant import Ramp, Constant
from hoomd.write import GSD

from ase import Atoms
from ase.data import atomic_numbers
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize

from Structure_recognition import CNA_Classification, PTM_Classification


class ProgressReporter(hoomd.custom.Action):
    def __init__(self, sim, total_steps, timestep0, report_interval=60):
        self.sim = sim
        self.total_steps = total_steps
        self.report_interval = report_interval
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.last_step = 0
        self.time_step = 0
        self.last_time = self.start_time
        self.timestep0 = timestep0

    def act(self, timestep):
        current_time = time.time()
        elapsed = current_time - self.start_time
        delta_t = current_time - self.last_time
        delta_steps = timestep - self.last_step
        tps = delta_steps / delta_t if delta_t > 0 else 0

        steps_remaining = self.total_steps - (timestep - self.timestep0)
        eta_seconds = float(steps_remaining / tps) if tps > 0 else 0

        if current_time - self.last_print_time >= self.report_interval:
            print(
                f"Time {str(timedelta(seconds=int(elapsed)))} | "
                f"Step {timestep - self.timestep0} / {self.total_steps} | "
                f"TPS {tps:.3f} | ETA {str(timedelta(seconds=int(eta_seconds)))}"
            )
            self.last_print_time = current_time

        self.last_step = timestep
        self.last_time = current_time


def load_goals(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    rdf_data = data['rdf']

    x_vals_AA = []
    x_vals_AB = []
    x_vals_BB = []
    y_vals_AA = []
    y_vals_AB = []
    y_vals_BB = []

    if "('A', 'A')" in rdf_data:
        for pair in rdf_data["('A', 'A')"]:
            x_vals_AA.append(pair[0])
            y_vals_AA.append(pair[1])
    if "('A', 'B')" in rdf_data:
        for pair in rdf_data["('A', 'B')"]:
            x_vals_AB.append(pair[0])
            y_vals_AB.append(pair[1])
    if "('B', 'B')" in rdf_data:
        for pair in rdf_data["('B', 'B')"]:
            x_vals_BB.append(pair[0])
            y_vals_BB.append(pair[1])

    return np.array(y_vals_AA), np.array(y_vals_AB), np.array(y_vals_BB)


def lj_nm_cut(r, sigma, lamda, n, m):
    if isinstance(r, torch.Tensor):
        r = r.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    if isinstance(lamda, torch.Tensor):
        lamda = lamda.detach().cpu().numpy()

    E0 = 1.0
    r = np.asarray(r)
    r_safe = np.where(r < 1e-8, 1e-8, r)

    sr_n = (sigma / r_safe) ** n
    sr_m = (sigma / r_safe) ** m

    u_lj = 4 * E0 * (sr_m - sr_n)
    f_lj = 4 * E0 * (m * sr_m - n * sr_n) / r_safe

    r_cut = sigma * 2 ** (1 / 6)

    u_att = np.where(r <= r_cut, u_lj + (1 - lamda) * E0, lamda * u_lj)
    f_att = np.where(r <= r_cut, f_lj, lamda * f_lj)

    return u_att, f_att


def compute_rdf_pair(query, positions_j, num_bins, r_max, box):
    rdf = freud.density.RDF(bins=num_bins, r_max=r_max)
    rdf.compute(system=(box, positions_j), query_points=query)
    bin_centers = rdf.bin_centers
    rdf_interv = rdf.rdf.copy()
    rdf_interv[bin_centers < 0.8] = 0
    return bin_centers, rdf_interv


def compute_rdf(gsd_file, r_lj_mn, output_svg, parameters):
    """
    Compute RDF from a GSD file using the last frame and r_lj_mn as the bin centers.

    Parameters:
        gsd_file (str): Path to the .gsd trajectory file.
        r_lj_mn (np.ndarray): Array of distances to set bin resolution and r_max.

    Returns:
        r (np.ndarray): Bin centers (r values).
        g_r (np.ndarray): RDF values.
    """
    num_bins = len(r_lj_mn)
    r_max = r_lj_mn[-1]

    try:
        pipeline = import_file(gsd_file)
    except Exception as e:
        print("Error importing file:", e)
        exit(1)

    frame_index = pipeline.source.num_frames - 1
    print("Frame index for g of r:", frame_index)

    try:
        frame = pipeline.compute(frame_index)
    except Exception as ee:
        print(f"Error processing frame {frame_index}: {ee}")
        return

    box_matrix = frame.cell.matrix
    if parameters.three_d == 1:
        freud_box = freud.box.Box.from_matrix(box_matrix)
    else:
        Lx = box_matrix[0][0]
        Ly = box_matrix[1][1]
        freud_box = freud.box.Box(Lx=Lx, Ly=Ly, is2D=True)

    positions = frame.particles["Position"].array
    typeid = frame.particles["Particle Type"].array
    types = frame.particles["Particle Type"].types

    try:
        type_A = 0
        type_B = 1
    except ValueError as e:
        print("Could not find particle types 'A' and 'B' in:", types)
        return

    pos_A = positions[typeid == type_A]
    pos_B = positions[typeid == type_B]

    path = os.path.dirname(gsd_file)
    r, g_aa = compute_rdf_pair(pos_A, pos_A, num_bins, r_max, box=freud_box)
    _, g_ab = compute_rdf_pair(pos_A, pos_B, num_bins, r_max, box=freud_box)
    _, g_bb = compute_rdf_pair(pos_B, pos_B, num_bins, r_max, box=freud_box)
    _, g_ba = compute_rdf_pair(pos_B, pos_A, num_bins, r_max, box=freud_box)

    return r, g_aa, g_ab, g_bb


def wait_until_ready(output_file_traj, max_attempts=10, delay=5):
    for attempt in range(max_attempts):
        try:
            pipeline = import_file(output_file_traj)
            frame_count = pipeline.source.num_frames
            print(f"Attempt {attempt+1}: Found {frame_count} frame(s).")
            if frame_count > 0:
                return pipeline
        except Exception as e:
            print(f"Attempt {attempt+1}: Error reading file {output_file_traj} - {e}")
        time.sleep(delay)
    raise RuntimeError("File not ready after multiple attempts.")


def callhoomd(state_history, directory, index_AA, index_ij, index_BB, gpu_id, parameters, use_slurm_gpu=False):
    import hoomd.md

    if not use_slurm_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[INFO] Manually setting GPU {gpu_id} via CUDA_VISIBLE_DEVICES")
    else:
        slurm_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
        print(f"[INFO] Using SLURM-assigned GPU: {slurm_gpu} (task gpu_id={gpu_id} for file naming)")

    os.environ["OMP_NUM_THREADS"] = "1"

    if hasattr(parameters, 'noise'):
        seed_assigned = parameters.str_index + gpu_id * 1000
        output_file_traj = os.path.join(directory, f'2D_sss_laa0_reproduce_{gpu_id}.gsd')
        frames = parameters.frames
    else:
        seed_assigned = parameters.str_index
        output_file_traj = os.path.join(directory, f'{parameters.structure}_{index_BB}_for_AA_{index_ij}_for_IJ_gpu_{gpu_id}.gsd')
        frames = 10

    output_file = parameters.system_file

    try:
        device = hoomd.device.GPU()
    except Exception as e:
        device = hoomd.device.CPU()

    sim = hoomd.Simulation(device=device, seed=seed_assigned)

    gpu_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")

    if isinstance(sim.device, hoomd.device.GPU):
        print(f"[INFO] HOOMD is running on the GPU: {gpu_visible}")
    else:
        print("[ERROR] HOOMD is NOT using the GPU backend!")

    omp_threads       = os.environ.get("OMP_NUM_THREADS", "NOT SET")
    actual_cpu_threads = multiprocessing.cpu_count()

    sim.create_state_from_gsd(filename=output_file, frame=-1)
    sim_timestep0 = sim.timestep
    print("Before reset:", sim.timestep)

    steps         = parameters.Nc
    delta_t       = parameters.dt
    limit_in_hours = parameters.hour_limit

    L0x = sim.state.box.Lx
    L0y = sim.state.box.Ly
    L0z = sim.state.box.Lz

    nl = Cell(buffer=0.4)

    rAA, UAA, fAA = np.loadtxt(os.path.join(directory, f"Potentials/attractive_forces_AA_{gpu_id}.dat"), unpack=True)
    rAB, UAB, fAB = np.loadtxt(os.path.join(directory, f"Potentials/attractive_forces_IJ_{gpu_id}.dat"), unpack=True)
    rBB, UBB, fBB = np.loadtxt(os.path.join(directory, f"Potentials/attractive_forces_BB_{gpu_id}.dat"), unpack=True)

    width = len(rAA)
    table = hoomd.md.pair.Table(nlist=nl)

    table.params[('A', 'A')] = {'r_min': rAA[0], 'U': UAA, 'F': fAA}
    table.params[('A', 'B')] = {'r_min': rAB[0], 'U': UAB, 'F': fAB}
    table.params[('B', 'B')] = {'r_min': rBB[0], 'U': UBB, 'F': fBB}
    table.r_cut[('A', 'A')] = rAA[-1]
    table.r_cut[('A', 'B')] = rAB[-1]
    table.r_cut[('B', 'B')] = rBB[-1]

    for pair in [('A', 'A'), ('A', 'B'), ('B', 'B')]:
        p = table.params[pair]
        assert 'r_min' in p and isinstance(p['r_min'], float)
        assert 'U' in p and isinstance(p['U'], np.ndarray)
        assert 'F' in p and isinstance(p['F'], np.ndarray)

    integrator = hoomd.md.Integrator(dt=delta_t, forces=[table])

    from hoomd.variant import Variant

    class TempSchedule(Variant):
        def __init__(self, t1, t2, t3):
            super().__init__()
            self.t1 = t1
            self.t2 = t2
            self.t3 = t3

        def __call__(self, timestep):
            t = timestep - sim_timestep0
            if t < self.t1:
                return 1.0 - 0.99 * (t / self.t1)
            else:
                return 0.01

        def _min(self):
            return 1.0

        def _max(self):
            return 3.0

    t1    = int(steps)
    t2    = int(2 / 6 * steps)
    t3    = int(steps) - t1 - t2
    steps = int(steps)

    kT_variant = TempSchedule(steps, t2, t3)

    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(),
        kT=kT_variant,
    )
    integrator.methods.append(langevin)

    sim.operations.integrator = integrator
    sim.operations.integrator.methods[0].gamma['A'] = 0.1
    sim.operations.integrator.methods[0].gamma['B'] = 0.1

    thermo_properties = md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo_properties)

    logger = hoomd.logging.Logger()
    logger.add(thermo_properties, quantities=['kinetic_temperature', 'pressure', 'volume', 'potential_energy'])

    gsd_writer = hoomd.write.GSD(
        filename=output_file_traj,
        trigger=hoomd.trigger.Periodic(period=int(steps / frames)),
        mode='ab',
        logger=logger,
        dynamic=['attribute', 'property'],
        filter=hoomd.filter.All(),
    )
    sim.operations.writers.append(gsd_writer)

    progress = ProgressReporter(sim=sim, total_steps=steps, timestep0=sim_timestep0, report_interval=60)
    progress_action = hoomd.update.CustomUpdater(
        action=progress,
        trigger=hoomd.trigger.Periodic(100),
    )
    sim.operations.updaters.append(progress_action)

    x_yield = None
    flag    = 1

    start_time = time.time()
    timestep0  = sim.timestep
    print("Time of simulation:", float(limit_in_hours * 3600))
    print("Timestep of simulation:", sim.timestep)

    try:
        sim.run(1)
        while time.time() - start_time < float(limit_in_hours * 3600) and sim.timestep < timestep0 + int(steps):
            sim.run(int(steps / 10 + 1))

        for writer in sim.operations.writers:
            if isinstance(writer, hoomd.write.GSD):
                writer.flush()
        flag = 0

        if parameters.three_d == 1:
            x_yield, flag = PTM_Classification(state_history, output_file_traj, parameters)
        else:
            print("Quantifying 2D structure...")
            x_yield, flag = CNA_Classification(output_file_traj, parameters)

    except Exception as e:
        print("Error -", e)
        print("Simulation failed - setting default values")
        x_yield = 0.0
        flag    = 1

    if x_yield is None:
        print("WARNING: x_yield was not set, using default value 0.0")
        x_yield = 0.0
        flag    = 1

    return x_yield, flag, output_file_traj


def run_step_parallel(state_history, sii, sij, lamda_AA, lamda_ij, lamda_BB, parameters, epoch, num_threads, gpu_id, use_slurm_gpu=False):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    if not use_slurm_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    r_lj_mn = np.arange(0.0, 5, 0.05)

    index_BB = f"{epoch}_{sii:.3f}_{lamda_BB:.3f}"
    index_ij = f"{sij:.3f}_{lamda_ij:.3f}"
    index_AA = f"{epoch}_{sii:.3f}_{lamda_AA:.3f}"

    u_att_AA, f_att_AA = lj_nm_cut(r_lj_mn, sii, lamda_AA, n=6, m=12)
    u_att_BB, f_att_BB = lj_nm_cut(r_lj_mn, sii, lamda_AA, n=6, m=12)
    u_rep,    f_rep    = lj_nm_cut(r_lj_mn, sij, lamda_ij, n=6, m=12)

    directory     = os.getcwd()
    save_dir      = os.path.join(directory, 'Trajectories_and_Potentials')
    potential_dir = os.path.join(directory, 'Trajectories_and_Potentials/Potentials')
    os.makedirs(potential_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(os.path.join(potential_dir, f"attractive_forces_AA_{gpu_id}.dat"), np.column_stack([r_lj_mn[1:], u_att_AA[1:], f_att_AA[1:]]))
    np.savetxt(os.path.join(potential_dir, f"attractive_forces_BB_{gpu_id}.dat"), np.column_stack([r_lj_mn[1:], u_att_BB[1:], f_att_BB[1:]]))
    np.savetxt(os.path.join(potential_dir, f"attractive_forces_IJ_{gpu_id}.dat"), np.column_stack([r_lj_mn[1:], u_rep[1:],    f_rep[1:]]))

    x_k, flag, output_file_traj = callhoomd(state_history, save_dir, index_AA, index_ij, index_BB, gpu_id, parameters, use_slurm_gpu)
    rdf, gAA, gAB, gBB = compute_rdf(output_file_traj, r_lj_mn, output_svg=np.array(["rdf_AA.svg", "rdf_AB.svg", "rdf_BB.svg"]), parameters=parameters)

    file_path = parameters.goal_rdf_json_file
    gaa_goal, gab_goal, gbb_goal = load_goals(file_path)

    xmax = 1.0
    if parameters.first_run == 1:
        condition = flag == 1 and epoch > 8 and epoch % 4 == 0
    else:
        condition = flag == 1 and epoch > 1 and epoch % 4 == 0

    goal_fraction = x_k[-1, parameters.ig]

    if condition:
        print("The actions predicted end up on chaotic system")
        value = 1.0
    else:
        value = 0.0 if goal_fraction >= parameters.success_threshold else 1.0

    Reward = value

    r_vals = r_lj_mn[:]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].fill_between(r_vals, gaa_goal, color='grey', alpha=0.5, label='Target AA RDF')
    axes[0].plot(r_vals, gAA, color='blue', marker='o', linestyle='--', label='Sampled AA RDF')
    axes[0].set_ylabel("RDF")
    axes[0].set_ylim(-0.1, 10)
    axes[0].set_title("AA RDF")
    axes[0].legend()
    axes[0].grid(False)

    axes[1].fill_between(r_vals, gab_goal, color='grey', alpha=0.5, label='Target AB RDF')
    axes[1].plot(r_vals, gAB, color='green', marker='o', linestyle='--', label='Sampled AB RDF')
    axes[1].set_ylabel("RDF")
    axes[1].set_ylim(-0.1, 10)
    axes[1].set_title("AB RDF")
    axes[1].legend()
    axes[1].grid(False)

    axes[2].fill_between(r_vals, gbb_goal, color='grey', alpha=0.5, label='Target BB RDF')
    axes[2].plot(r_vals, gBB, color='red', marker='o', linestyle='--', label='Sampled BB RDF')
    axes[2].set_xlabel("Distance r")
    axes[2].set_ylabel("RDF")
    axes[2].set_ylim(-0.1, 10)
    axes[2].set_title("BB RDF")
    axes[2].legend()
    axes[2].grid(False)

    plt.tight_layout()
    
    os.makedirs("RDF_plots", exist_ok=True)
    plt.savefig(f"RDF_plots/rdf_all_subplots_{epoch}_{gpu_id}.svg")

    print("X_k returned", x_k[-1])
    print("x_k whole", x_k)
    print("Reward signal", -Reward)

    return x_k, -Reward


def get_slurm_nodes():
    """
    Get list of nodes allocated by SLURM.
    Returns list of node names, or None if not in SLURM environment.

    If HEALTHY_NODES_OVERRIDE environment variable is set (by health check script),
    uses only those nodes instead of full SLURM allocation.
    """
    import re

    healthy_override = os.environ.get('HEALTHY_NODES_OVERRIDE')
    if healthy_override:
        nodes = [n.strip() for n in healthy_override.split(',') if n.strip()]
        print(f"[INFO] Using HEALTHY_NODES_OVERRIDE: {nodes}")
        return nodes

    nodelist = os.environ.get('SLURM_JOB_NODELIST') or os.environ.get('SLURM_NODELIST')
    if not nodelist:
        return None

    print(f"[INFO] SLURM nodelist: {nodelist}")

    try:
        result = subprocess.run(
            ['scontrol', 'show', 'hostnames', nodelist],
            capture_output=True, text=True, check=True,
        )
        nodes = result.stdout.strip().split('\n')
        print(f"[INFO] Expanded nodes: {nodes}")
        return nodes
    except Exception as e:
        print(f"[WARNING] Could not parse SLURM nodelist: {e}")
        nodes = [n.strip() for n in nodelist.split(',')]
        return nodes


def run_multinode_slurm(args_list, nodes, parameters):
    """
    Execute GPU tasks across multiple SLURM nodes using srun.

    Args:
        args_list: List of argument tuples for run_step_parallel
        nodes: List of node names from SLURM
        parameters: Parameters object

    Returns:
        List of results from all tasks
    """
    batch_size = len(args_list)
    num_nodes  = len(nodes)

    result_dir = os.path.join(os.getcwd(), "multinode_results")
    os.makedirs(result_dir, exist_ok=True)

    wrapper_script = os.path.join(result_dir, "gpu_task_wrapper.py")
    with open(wrapper_script, 'w') as f:
        f.write("""#!/usr/bin/env python
import sys
import pickle
import os

# Add parent directory to path to import MD_engine
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
""")

    os.chmod(wrapper_script, 0o755)

    processes  = []
    task_files = []

    node_gpu_counter = {node: 0 for node in nodes}

    for i, args in enumerate(args_list):
        node_idx     = i % num_nodes
        target_node  = nodes[node_idx]
        local_gpu_id = node_gpu_counter[target_node]
        node_gpu_counter[target_node] += 1

        args_file   = os.path.join(result_dir, f"task_{i}_args.pkl")
        result_file = os.path.join(result_dir, f"task_{i}_result.pkl")

        with open(args_file, 'wb') as f:
            pickle.dump(args, f)

        task_files.append((args_file, result_file))

        srun_cmd = [
            'srun',
            '--nodes=1',
            '--ntasks=1',
            '--nodelist', target_node,
            '--overlap',
            'python', wrapper_script, args_file, result_file, str(local_gpu_id),
        ]

        print(f"[INFO] Launching task {i} on node {target_node} (local GPU {local_gpu_id})")
        print(f"[CMD] {' '.join(srun_cmd)}")

        proc = subprocess.Popen(
            srun_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((proc, i, target_node))

    print(f"[INFO] Waiting for {len(processes)} tasks to complete in parallel...")

    active_processes = list(processes)
    completed = []

    while active_processes:
        for proc, task_id, node in active_processes[:]:
            retcode = proc.poll()
            if retcode is not None:
                stdout, stderr = proc.communicate()
                completed.append((proc, task_id, node, stdout, stderr))
                active_processes.remove((proc, task_id, node))
                print(f"[INFO] Task {task_id} on {node} finished (return code: {retcode})")

        if active_processes:
            time.sleep(0.5)

    print(f"[INFO] All {len(processes)} tasks completed. Printing outputs...")
    for proc, task_id, node, stdout, stderr in completed:
        if proc.returncode != 0:
            print(f"[WARNING] Task {task_id} on {node} failed with return code {proc.returncode}")
            print(f"[STDERR] {stderr}")
        else:
            print(f"[SUCCESS] Task {task_id} on {node} completed")

        if stdout:
            print(f"[STDOUT Task {task_id}] {stdout}")

    results = []
    for i, (args_file, result_file) in enumerate(task_files):
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            results.append(result)
            print(f"[INFO] Loaded result for task {i}")
        else:
            print(f"[ERROR] Result file missing for task {i}, using default")
            results.append((0.0, -1.0))

        try:
            os.remove(args_file)
            os.remove(result_file)
        except:
            pass

    return results


def calc_state(state, actions, batch_size, parameters, state_history, epoch):
    u = actions.detach().cpu()
    print("Actions:", actions)

    sii       = u[:, 0]
    sij       = u[:, 1]
    lambda_ii = u[:, 2]
    lambda_ij = u[:, 3]

    num_threads_per_run = 1
    args_list = [
        (state_history, sii[i], sij[i], lambda_ii[i], lambda_ij[i], lambda_ii[i], parameters, epoch, num_threads_per_run, i)
        for i in range(batch_size)
    ]

    nodes = get_slurm_nodes()
    use_multinode = nodes is not None and len(nodes) > 1

    if use_multinode:
        print(f"[INFO] Multi-node execution detected: {len(nodes)} nodes")
        print(f"[INFO] Distributing {batch_size} GPU tasks across nodes: {nodes}")
        results = run_multinode_slurm(args_list, nodes, parameters)
    else:
        print(f"[INFO] Single-node execution")
        num_cores   = multiprocessing.cpu_count()
        max_workers = max(1, num_cores // num_threads_per_run)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_step_parallel, *args) for args in args_list]
            results = [f.result() for f in futures]

    state_vec  = []
    reward_vec = []
    for res in results:
        if isinstance(res, dict):
            state_part  = res.get("state", res.get("x_k", res))
            reward_part = res.get("reward", res.get("Reward", res))
        else:
            try:
                state_part, reward_part = res
            except (TypeError, ValueError):
                state_part, reward_part = res, res

        obs_dim = parameters.obs_dim if hasattr(parameters, 'obs_dim') else state.shape[-1]
        if isinstance(state_part, (int, float)):
            # error fallback — expand scalar to zero vector so buffer shape is correct
            state_val = np.zeros(obs_dim, dtype=np.float32)
        else:
            arr = np.asarray(state_part, dtype=np.float32)
            if arr.ndim == 2:
                state_val = arr[-1]   # (T, obs_dim) → take last timestep
            else:
                state_val = arr       # already (obs_dim,)

        state_vec.append(state_val)
        reward_vec.append(reward_part)

    next_state = torch.tensor(np.stack(state_vec, axis=0), dtype=torch.float32)
    reward     = torch.tensor([x for x in reward_vec], dtype=torch.float32)

    print("Next state:", next_state)
    print("Reward:", reward)

    return next_state, reward

