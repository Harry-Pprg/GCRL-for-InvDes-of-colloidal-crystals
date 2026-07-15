# GCRL for Inverse Design of Colloidal Crystals

**From Disorder to Crystal: Goal-Conditioned Reinforcement Learning for Inverse Design of Binary Colloidal Structures**

Harry Papargyriou¹, Ryan Soucek², Hsiao-Yen Beth Wei², Jeetain Mittal²·³·⁴, Ali Mesbah¹·*

¹ Department of Chemical and Biomolecular Engineering, University of California, Berkeley  
² Artie McFerrin Department of Chemical Engineering, Texas A&M University  
³ Department of Chemistry, Texas A&M University  
⁴ Interdisciplinary Graduate Program in Genetics and Genomics, Texas A&M University  

*Preprint — July 2026*

---

## Table of Contents

- [Abstract](#abstract)
- [Framework](#framework)
- [Key Results](#key-results)
- [Method Summary](#method-summary)
  - [Inverse Design as a GCRL Problem](#inverse-design-as-a-gcrl-problem)
  - [Algorithm](#algorithm)
  - [Network Architecture](#network-architecture)
  - [Molecular Dynamics Protocol](#molecular-dynamics-protocol)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [HOOMD-blue Installation](#hoomd-blue-installation)
  - [OVITO Installation](#ovito-installation)
- [Usage](#usage)
  - [Basic Training](#basic-training)
  - [Configuration](#configuration)
  - [Output Files](#output-files)
  - [Post-Training Analysis](#post-training-analysis)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Workflow](#workflow)
- [System Requirements](#system-requirements)
- [Important Notes](#important-notes)
- [Multi-Node GPU Execution](#multi-node-gpu-execution)
- [Troubleshooting](#troubleshooting)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [Contact](#contact)

---

## Abstract

Colloidal self-assembly provides a scalable route to fabricating materials with emergent, structure-encoded properties. However, engineering pair potentials that realize target crystal motifs remains difficult as the interaction parameter space is highly non-linear, successful crystallization pathways are sparsely distributed within the design space, and self-assembly is co-governed by thermodynamic and kinetic accessibility. We present a goal-conditioned reinforcement learning (GCRL) framework that maps target crystal structures to experimentally motivated modified Lennard-Jones pair-potentials, inspired by DNA-functionalized particle interactions, where success of self-assembly is evaluated solely on whether the target structures are realized. The proposed GCRL framework relies on hindsight experience replay, which allows the agent to learn from all simulation outcomes, including runs that produce non-target structures. GCRL is validated on seven binary colloidal target crystal structures spanning 2D and 3D motifs — square single stripe, binary triangular kagome, square lattice, open honeycomb, simple cubic, cubic diamond and body-centered cubic — with the first two being previously realized only with purely repulsive potentials. GCRL does not require physically informed initialization of interaction parameters. By restricting pair potentials to single-well modified LJ forms while operating on a parameter space with sparse crystallization events, GCRL brings inverse design closer to the laboratory bottom-up synthesis of colloidal superlattices.

---

## Framework

![GCRL Framework](assets/GCRL_framework.png)

The closed-loop inverse design framework starts from a target crystal structure **g** and a disordered particle configuration **s**. The goal-conditioned RL policy π_θ(**α** | **s**, **g**) proposes an action **α** = [σ_AA/BB, λA_A/BB, λ_AB] consisting of modified Lennard-Jones pair potential parameters for like- and unlike-particle species. These potentials are evaluated through coarse-grained implicit-solvent MD simulations (HOOMD-blue, Langevin dynamics) under a temperature-annealing schedule (T* = 1 → 0.01). The final assembled configuration s′ is structurally analyzed and converted to a binary reward signal r ∈ {0, −1}. Transitions (s, α, r, s′, g) are stored in a replay buffer; hindsight experience replay (HER) relabels trajectories where non-target structures form, so that failed episodes still contribute useful learning signal.

---

## Key Results

- **7 binary colloidal crystal structures** successfully inverse-designed, spanning 2D and 3D motifs:
  - *2D*: Square Lattice (SL), Open Honeycomb (OHC), Square Single Stripe (SSS), Binary Triangular Kagome (BTr)
  - *3D*: Simple Cubic (SC), Cubic Diamond (CD), Body-Centered Cubic (BCC)
- SSS and BTr were **previously unrealized with attractive pair potentials** (bonding minima V < 0); GCRL designs enthalpically-driven interactions for both.
- GCRL **converges within ~50 training epochs** from a physics-agnostic initialization at the center of parameter space.
- The framework is **robust to initialization**: three independent runs for OHC starting from physically distinct regions of parameter space all converge to potentials that drive successful assembly.
- Validation simulations (10× longer, 3–4k particles) confirm reproducible crystallization across 5 independent stochastic trajectories per structure, with crystallization fractions of 0.8–1.0.
- The ratio $\sigma_{ii}/\sigma_{ij}$ emerges as the **primary geometric design parameter**; the well-depth ratio $\lambda_{ii}/\lambda_{ij}$ spans a broad basin of solutions, revealing a many-to-one mapping between pair potentials and assembled structures.

---

## Method Summary

### Inverse Design as a GCRL Problem

The interaction parameter vector $\mathbf{a} = [\sigma_{AA}, \sigma_{AB}, \lambda_{AA}, \lambda_{AB}]^\top \in \mathbb{R}^4$ defines modified Lennard-Jones potentials:

```math
u_{ij}(r) = \begin{cases}
u^{\text{LJ}}_{ij}(r) + (1 - \lambda_{ij})\,\varepsilon & r \leq 2^{1/6}\sigma_{ij} \\
\lambda_{ij}\, u^{\text{LJ}}_{ij}(r) & r > 2^{1/6}\sigma_{ij}
\end{cases}
```

where $u^{\text{LJ}}_{ij}(r) = 4\varepsilon\!\left[\left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^{6}\right]$ is the standard LJ potential. Setting $\lambda_{ij} = 0$ recovers the purely repulsive WCA potential; $\lambda_{ij} > 0$ introduces an attractive well, making the interaction enthalpically driven. The unlike-particle interaction length is fixed at $\sigma_{AB} = 1.0$.

The ID problem is cast as maximizing the probability that the assembled configuration is classified as the desired crystal motif $\mathbf{g}$:

```math
\mathbf{a} \in \underset{\mathbf{a} \in \mathcal{A}}{\arg\max}\; \mathbb{P}\!\left(\Psi(\Phi(x_f(\mathbf{a}))) = \mathbf{g}\right)
```

### Algorithm

The policy $\pi_\theta$ is a deterministic actor network $\mu_\theta: \mathcal{S} \times \mathcal{G} \rightarrow \mathbb{R}^{d_a}$ with externalized exploration variance $\sigma_t$ (performance-based noise scheduling). Value estimation uses twin critics $Q_{\phi_1}, Q_{\phi_2}$ (TD3-style clipped double Q-learning). The full algorithm combines:

- **Soft Actor-Critic (SAC)** with maximum-entropy objective and entropy scheduling ($\alpha_t$ decays linearly over $T_{\text{decay}}$ epochs)
- **Hindsight Experience Replay (HER)**: each transition stored twice — under the original goal $\mathbf{g}$ and under the actually achieved structure $\mathbf{g}'$ — densifying the reward distribution in a sparse crystallization landscape
- **Performance-based noise scheduling**: exploration variance $\sigma_t$ scales with empirical batch success rate $p_t$, maintaining high exploration until the policy achieves consistent success across the full batch:

```math
\boldsymbol{\sigma}_{t+1} = \begin{cases}
\sigma_{\text{hi}}\,\Delta\mathbf{a} & t < e_{\text{explore}} \\
\left[\Delta\sigma\,(1 - p_t)^B + \sigma_{\text{lo}}\right]\Delta\mathbf{a} & \text{otherwise}
\end{cases}
```

where $\sigma_{\text{hi}} = 0.45$, $\sigma_{\text{lo}} = 0.005$, $B = 6$, and $\Delta\mathbf{a} = \mathbf{a}_{\max} - \mathbf{a}_{\min}$.

### Network Architecture

Actor $\mu_\theta$ and twin critics $Q_{\phi_1}, Q_{\phi_2}$ are fully connected networks with two hidden layers of width 32, preceded by a bottleneck layer of width 20, and Tanh activations. Both actor and critics are optimized with Adam (actor lr $= 3\times10^{-3}$, critics lr $= 3\times10^{-4}$). Per training epoch: 10 critic gradient steps, then 1 actor update.

### Molecular Dynamics Protocol

- **Backend**: HOOMD-blue 5.2.0, Langevin dynamics, $\Delta t = 0.001\,\tau$, $\gamma = 0.1\,m/\tau$
- **Annealing**: $T^* = k_B T/\varepsilon$, cooled from $T^* = 1$ to $T^* = 0.01$
- **Training runs**: ~500 particles, $10^4\tau$ per episode
- **Validation runs**: 3000–4096 particles, $10^5\tau$, 5 independent trajectories per structure
- **Structure classification**: OVITO PTM and IDS — Identify Diamond Structure (3D systems), CNA (2D systems), cluster analysis via connected components

---

## GC_Std_scheduler

Reinforcement learning framework for designing interaction potentials to drive self-assembly of particle systems into target crystal structures.

## Overview

This project implements a reinforcement learning approach using a Goal-Conditioned Reinforcement Learning algorithm to learn optimal Lennard-Jones-type interaction potentials that guide particle systems toward specific target structures (e.g., FCC, HCP, Square Lattice, Honeycomb). The system uses molecular dynamics simulations via HOOMD-blue and structural analysis with OVITO's Polyhedral Template Matching (PTM).

## Key Features

- **Reinforcement Learning**: Policy gradient methods for learning optimal interaction parameters
- **Molecular Dynamics**: HOOMD-blue integration for efficient particle simulations
- **Structure Analysis**: Automated crystal structure classification using OVITO PTM and CNA
- **Parallel Computing**: Multi-threaded simulation execution for batch training
- **Experience Replay**: Buffer system for storing and replaying training trajectories
- **Visualization**: Comprehensive plotting of training metrics, success rates, and action evolution

## Project Structure

```
GC_Std_scheduler/
├── main_val.py                    # Main training script with NN architecture
├── reproduce_pot.py               # Parallel MD simulation engine
├── Buffer.py                      # Experience replay buffer with plotting methods
├── helper.py                      # Pretraining and visualization utilities
├── visualize.py                   # Training visualizer
├── cna_ohc_traj.py               # Structure analysis (CNA/PTM) for 2D systems
├── 2d_initial_conf.py            # 2D initial configuration generator
├── relabel_particles.py          # Particle relabeling utilities
├── crystal.conf                  # Configuration file for 2D structure analysis
├── Initial_Configurations/       # Initial system configurations (2D/3D)
├── Target_RDF/                   # Reference radial distribution functions
├── Model_and_Results/            # Trained models and training results
├── Trajectories_and_Potentials/  # Simulation outputs and interaction tables
├── RDF_plots/                    # RDF visualization outputs
└── PLOTTING_USAGE.md             # Documentation for Buffer plotting methods
```

## Core Components

### 1. Neural Network Architecture (main_val.py)

- **Policy Network**: Gaussian policy that outputs interaction potential parameters
- **Action Space**: Lennard-Jones parameters (σ, λ) for different particle pair types
- **Observation Space**: Structure fractions from PTM classification
- **Goal-Conditioned**: Networks receive both state and goal structure target

### 2. Simulation Engine (reproduce_pot.py)

- **MD Backend**: HOOMD-blue for GPU-accelerated molecular dynamics
- **Custom Potentials**: Tabulated pair potentials with trainable parameters
- **Structure Analysis**: Real-time PTM classification during simulation
- **Progress Tracking**: ETA estimation and throughput monitoring

### 3. Experience Buffer (Buffer.py)

Stores transitions (s, a, r, s', g) and provides:
- CSV-based persistence
- Success rate tracking per goal structure
- Action uncertainty analysis (exploration → exploitation)
- Q-value progression estimation

See `PLOTTING_USAGE.md` for detailed plotting API documentation.

### 4. Structure Analysis

- **3D Systems**: PTM classification (FCC, HCP, BCC, ICO, SC, Cubic/Hex Diamond)
- **2D Systems**: CNA-based classification (Square Lattice, Honeycomb/Graphene)
- **Metrics**: Normalized structure fractions relative to target goals

## Installation

### Prerequisites

```bash
# Core dependencies
conda install numpy matplotlib scipy
conda install freud-analysis
conda install ase dscribe scikit-learn
conda install -c conda-forge pandas=2.2.3

conda install -c conda-forge pytorch=2.7.0
conda install -c conda-forge freud=3.3.1
conda install -c conda-forge gsd=3.4.2

```

### HOOMD-blue Installation

```bash
# For GPU support (recommended)
conda install "hoomd=5.2.0=*gpu*" "cuda-version=12.6" -c conda-forge

# For CPU only
pip install hoomd
```

### OVITO Installation

```bash
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.12.4

```

## Usage

### Basic Training

```python
python main_val.py
```

Training parameters are configured within the script via an `args` namespace containing:
- `batch`: Batch size (number of parallel simulations)
- `obs_dim`: Observation dimension (number of structure types)
- `action_dim`: Action dimension (number of potential parameters)
- `goal_dic`: Dictionary mapping structure names to target fractions
- `ig`: Index of primary goal structure

### Configuration

Key parameters to adjust:

1. **Target Structure** (in `main_val.py`):
```python
args.goal_dic = {'FCC': 0.8, 'HCP': 0.1, 'BCC': 0.0, ...}
args.ig = 0  # Index of primary goal (FCC in this case)
```

2. **Simulation Settings** (in `reproduce_pot.py`):
```python
total_steps = 10000  # MD steps per episode
kT = 1.0            # Temperature
dt = 0.005          # Integration timestep
```

3. **Action Bounds**:
```python
u_min = [0.5, 0.0, 0.0]  # [σ_min, λ_min, λ_min]
u_max = [2.0, 2.0, 2.0]  # [σ_max, λ_max, λ_max]
```

### Output Files

Each training run creates a timestamped directory in `Model_and_Results/` containing:

```
Model_and_Results/YYYY-MM-DD_HH-MM-SS/
├── buffer.csv                    # Experience replay data
├── training_data.csv             # Per-epoch metrics
├── actions.csv                   # Action history
├── actor_final.pth               # Trained policy network
├── Learn_loss_plot.svg           # Loss over epochs
├── Learn_state_plot.svg          # Structure fraction over epochs
├── Learn_action_plot.svg         # Action evolution over epochs
├── success_rate_goal_X.svg       # Success rate for target structure
├── action_uncertainty.svg        # Action std over epochs
└── q_value_progression.svg       # Mean reward progression
```

### Post-Training Analysis

Generate plots from saved buffer:

```python
from Buffer import Buffer

buffer = Buffer(
    buffer_path="Model_and_Results/run_dir/buffer.csv",
    obs_dim=8, action_dim=4, u_min=[0.5,0,0], u_max=[2,1,1]
)

buffer.plot_success_rate(target_goal=0, batch_size=16, output_dir="plots/")
buffer.plot_action_uncertainty(output_dir="plots/")
buffer.plot_q_value_progression(output_dir="plots/")
```

## Workflow

1. **Initialization**: Generate or load initial particle configuration
2. **Policy Sampling**: Neural network samples interaction potential parameters
3. **MD Simulation**: HOOMD-blue runs simulation with custom potential
4. **Structure Analysis**: OVITO PTM classifies final particle structures
5. **Reward Calculation**: Compute reward based on target structure fraction
6. **Policy Update**: REINFORCE gradient update to maximize expected reward
7. **Repeat**: Continue for specified number of epochs

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended for HOOMD-blue)
- **Memory**: 8GB+ RAM for typical batch sizes
- **Storage**: ~1-10GB per training run depending on trajectory logging

## Important Notes

- **2D vs 3D**: System dimensionality is configured via `cna_ohc_traj.py` for 2D systems
- **Pretraining**: Use `helper.train_actor_to_target()` to initialize policy near known good actions
- **Checkpoints**: Models saved as `.pth` files can be reloaded for continued training
- **Validation**: Use `reproduce_pot.py` independently to validate learned potentials

## Multi-Node GPU Execution

For large-scale training, the code supports distributing simulations across multiple SLURM nodes. The `calc_state()` function automatically detects the environment and switches between single- and multi-node execution — no code changes required. See [`MULTI_NODE_SETUP.md`](MULTI_NODE_SETUP.md) for full details.

### How It Works

- **Single node** (`--nodes=1`): uses `ProcessPoolExecutor`, assigns `CUDA_VISIBLE_DEVICES` directly from `gpu_id`
- **Multi-node** (`--nodes=2+`): uses `srun` with round-robin task distribution across nodes; physical GPU assignment is handled automatically by SLURM via `--gpus-per-task=1`

Note: `gpu_id` (0 to `batch_size`−1) is used only for file naming and random seed generation — not for physical GPU selection in multi-node mode.

### SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=multi_node_job
#SBATCH --partition=gpu
#SBATCH --nodes=2                    # increase for multi-node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4            # 4 GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --time=40:30:00
#SBATCH --output=train_out_%j.out
#SBATCH --error=train_err_%j.err

python -u main_val.py \
  --epochs 110 --batch 4 --first_run 1 --Nc 10000000 \
  --sigma_hi 0.40 --sigma_lo 0.005 --str_index 5 --three_d 1 \
  --pretrain_target 1.00 0.5 0.5 --success_threshold 0.6
```

With 2 nodes × 4 GPUs, a batch of 8 simulations runs fully in parallel (~2× speedup over single node).

### Temporary Files

During multi-node runs, the code creates a `multinode_results/` directory for inter-process coordination (pickle files for task args and results). These are cleaned up automatically after each epoch.

### Monitoring

```bash
# Stream training output
tail -f train_out_<jobid>.out

# Check GPU usage interactively
srun --jobid=<jobid> --pty bash
nvidia-smi
```

Expected log output:
```
[INFO] Multi-node execution detected: 2 nodes
[INFO] Distributing 8 GPU tasks across nodes: ['node001', 'node002']
[INFO] Launching task 0 on node node001 (GPU 0)
[SUCCESS] Task 0 on node001 completed
```

### Multi-Node Troubleshooting

| Issue | Solution |
|---|---|
| Tasks not distributing | Check `echo $SLURM_NODELIST` is set and `--nodes=2` is in the script |
| GPU conflicts | Code uses `--exclusive` flag in `srun` automatically |
| File not found errors | Ensure all nodes share the same filesystem |
| Still using single-node | Verify allocated nodes with `scontrol show job <jobid>` |

---

## Troubleshooting

### Common Issues

1. **HOOMD-blue GPU errors**: Ensure CUDA-compatible GPU and drivers installed
2. **OVITO import errors**: May require system-specific OVITO installation
3. **Memory overflow**: Reduce batch size or trajectory length
4. **Convergence issues**: Try pretraining or adjusting learning rate

### Performance Optimization

- Enable GPU acceleration in HOOMD-blue for 10-100x speedup
- Adjust `ProcessPoolExecutor` workers based on CPU core count
- Use trajectory downsampling for structure analysis
- Consider multi-GPU setups for large batch sizes

## Data Availability

Reference data for all seven studied crystal structures is included in this repository:

| Directory / File | Contents |
|---|---|
| `Initial_Configurations/` | Thermalized disordered initial particle configurations used for both training and validation runs (2D and 3D systems) |
| `Target_RDF/` | Target radial distribution functions (RDFs) for each crystal structure, used as reference descriptors during training |
| `crystal.conf` | CNA descriptor definitions for 2D structure classification (Square Lattice, Honeycomb, Square Single Stripe, Binary Triangular Kagome) |
| `2d_initial_conf.py` | Script to generate perfect crystal configurations of the target structures, which can be used as reference or to create new initial conditions |

These files are sufficient to reproduce all training runs reported in the paper without generating new initial conditions.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{papargyriou2026gcrl,
  title   = {From Disorder to Crystal: Goal-Conditioned Reinforcement Learning
             for Inverse Design of Binary Colloidal Structures},
  author  = {Papargyriou, Harry and Soucek, Ryan and Wei, Hsiao-Yen Beth
             and Mittal, Jeetain and Mesbah, Ali},
  year    = {2026},
  note    = {Preprint}
}
```

## Contact

- **Code & implementation questions**: Harry Papargyriou — h.pap@berkeley.edu
- **Scientific & paper questions**: Ali Mesbah (corresponding author) — mesbah@berkeley.edu

## Acknowledgments

- HOOMD-blue development team
- OVITO development team
- PyTorch team
