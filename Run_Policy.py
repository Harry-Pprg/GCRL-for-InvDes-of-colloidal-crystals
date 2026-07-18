import sys
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.distributions import Normal

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting

from ovito.modifiers import *
from ovito.vis import *





from MD_engine import calc_state  
from Replay_Buffer import Buffer

def inverse_transform(actions, u_min, u_max):
    """
    Convert bounded actions back to pre-tanh space for pretraining.

    Args:
        actions: [batch_size, action_dim] in [u_min, u_max]
        u_min, u_max: action bounds

    Returns:
        pre_tanh: [batch_size, action_dim] in unbounded space
    """
    scale = u_max - u_min
    ratio = (actions - u_min) / scale  # Map to [0, 1]
    tanh_output = ratio * 2.0 - 1.0    # Map to [-1, 1]

    # Inverse tanh (arctanh), clamped for numerical stability
    pre_tanh = torch.arctanh(torch.clamp(tanh_output, -0.99, 0.99))
    return pre_tanh


def pretrain_policy_mean(policy, target_actions, states, goal_template, device, lr=1.5e-2, num_steps=250):
    """
    Pretrain policy μ to output target_actions for given states.

    Args:
        policy: ActorNet instance
        target_actions: [batch_size, action_dim] - desired mean actions
        states: [batch_size, obs_dim] - initial states
        goal_template: [obs_dim] - goal one-hot template (will be expanded to batch)
        device: torch device
        lr: learning rate for pretraining
        num_steps: number of gradient steps

    Returns:
        policy: pretrained policy
    """
    print("\n" + "="*70)
    print("PRETRAINING POLICY TO INITIAL CONDITIONS")
    print("="*70)
    print(f"Target actions: {target_actions[0].cpu().numpy()}")
    print(f"Learning rate: {lr}, Steps: {num_steps}")

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Convert target to tensors on device
    target_actions = target_actions.to(device)
    states = states.to(device)
    goal_template = goal_template.to(device)

    # Expand goal template to match batch size
    batch_size = states.shape[0]
    goal_vector = goal_template.unsqueeze(0).expand(batch_size, -1)

    # Convert target actions to pre-tanh space
    target_mu = inverse_transform(target_actions, policy.u_min.to(device), policy.u_max.to(device))

    for step in range(num_steps):
        sg = torch.cat((states, goal_vector), dim=1)

        # Get current μ (raw ne
        # rk output before tanh)
        mu = policy.net(sg)

        # MSE loss between current and target μ
        loss = loss_fn(mu, target_mu)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 20 == 0 or step == num_steps - 1:
            # Show actual actions (after tanh transform)
            with torch.no_grad():
                scale = policy.u_max.to(device) - policy.u_min.to(device)
                current_actions = torch.mul((torch.tanh(mu) + 1.0) / 2.0, scale) + policy.u_min.to(device)
                print(f"  Step {step:3d}: loss={loss.item():.6f} | actions={current_actions[0].cpu().numpy()}")

    print("Pretraining complete!")
    print("="*70 + "\n")
    return policy


def compute_goal_reward(next_state_fractions, goal_idx, success_threshold=0.8):
    """
    Compute sparse binary reward for a specific goal structure.

    Standard HER implementation: 0 if goal achieved, -1 otherwise.

    Args:
        next_state_fractions: numpy array or tensor [obs_dim] with structure fractions (NORMALIZED)
        goal_idx: int, index of target structure (0-8)
        success_threshold: float, threshold for success (default 0.8)

    Returns:
        reward: float, 0 if normalized fraction >= success_threshold, else -1
    """
    if isinstance(next_state_fractions, torch.Tensor):
        next_state_fractions = next_state_fractions.cpu().numpy()

    # Extract fraction of goal structure (NORMALIZED by goal_dic)
    goal_fraction = next_state_fractions[goal_idx]

    # Sparse binary reward (standard HER)
    reward = 0.0 if goal_fraction >= success_threshold else -1.0
    return reward

def alpha_scheduler(epoch):
    """
    Simple linear decay scheduler for alpha (temperature) parameter.

    Args:
        epoch: current epoch number
    """
    initial_alpha = 1.0
    final_alpha = 0.00
    total_epochs = 10

    if epoch >= total_epochs:
        return final_alpha
    else:
        alpha = initial_alpha - (initial_alpha - final_alpha) * (epoch / total_epochs)
        return alpha

class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim, 
                 batch_size ,
                 u_min=torch.tensor([0.4, 0.0, 0.0]),
                 u_max=torch.tensor([2.5, 3.0, 3.0]),
                 ):
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*obs_dim, 20),  #sg input layer
            nn.Tanh(),
            nn.Linear(20, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim) #mu only (std comes from scheduler)

        )
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.u_min = u_min
        self.u_max = u_max
        self.batch_size = batch_size

    def forward(self, x, sigma, return_dist_params=False):
        mu = self.net(x)  # Only predict mean
        scale = (self.u_max - self.u_min)

        # Use scheduled sigma instead of predicted std
        dist = Normal(mu, sigma)
        action_ratio = dist.rsample()
        actual_batch_size = mu.shape[0]
        sigma_ab = torch.ones(actual_batch_size, dtype=torch.float32, device=mu.device)
        bound_action =  torch.mul((torch.tanh(action_ratio) + 1.0) / 2.0, scale) + self.u_min
        action = torch.stack([bound_action[:,0], sigma_ab, bound_action[:,-2], bound_action[:,-1]], dim=1)

        # Log probability with tanh Jacobian correction
        log_prob = dist.log_prob(action_ratio).sum(dim=-1)  # [batch_size]

        # Add small epsilon for numerical stability
        tanh_correction = torch.log(1 - torch.tanh(action_ratio)**2 + 1e-6).sum(dim=-1)
        log_prob = log_prob - tanh_correction

        if return_dist_params:
            return action, log_prob, mu
        return action, log_prob
    
class CriticNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*obs_dim + action_dim , 20),  #sga input layer
            nn.Tanh(),
            nn.Linear(20, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1) #mu and std for each action dim
            
        )
        self.action_dim = action_dim
        self.obs_dim = obs_dim
    
    def forward(self, x):
        return self.net(x)  # bound Q in [0, 1]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Accept individual array as input")
    ## Boolean parameters
    parser.add_argument('--first_run', type=int, default = 1, help="First run boolean variable: 0 or 1")
    parser.add_argument('--novelty',type=int,default=0, help= "Flag so we can apply Novelty Search instead of Objective Search")

    ## Training parameters
    parser.add_argument('--batch', type = int, default = 6, help="Batch size")
    parser.add_argument('--epochs', type = int, default = 200, help="Epochs size")
    parser.add_argument('--actor_lr', type=float, default=3e-3, help='Learning rate for Actor network')
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Learning rate for Critic networks')
    parser.add_argument('--alpha_lr', type=float, default=3e-3, help='Learning rate for Alpha (temperature) parameter - increased for faster exploitation')

    ## Policy Gradient parameters
    # parser.add_argument('--sigma_perc', type = float, nargs = 2, default = [0.15, 0.01], help="Sigma value")
    parser.add_argument('--sigma_hi', type=float, default=0.40, help='Upper bound for noise scheduler (exploration)')
    parser.add_argument('--sigma_lo', type=float, default=0.005, help='Lower bound for noise scheduler (exploitation)')
    # parser.add_argument('--lr_boost_multiplier', type=float, default=10.0, help='LR multiplier when training on successful transitions (reward=0)')
    parser.add_argument('--success_threshold', type=float, default=0.6, help='Threshold for considering a structure formation successful (goal fraction >= threshold)')
    parser.add_argument('--exploration_epochs', type=int, default=20, help='Number of epochs to explore all goals before filtering to target goal only')
    parser.add_argument('--pretrain_mu', type=int, default=1, help='Pretrain policy mean to initial conditions (0=disable, 1=enable)')
    parser.add_argument('--pretrain_target', type=float, nargs=3, default=[1.45, 1.5, 1.5], help='Target actions for pretraining [sigma_II, lambda_II, lambda_IJ]')
    parser.add_argument('--action_dim', type = int, default = 3, help="Action dimensions")
    parser.add_argument('--obs_dim',type = int,  help="Observation Dimensions (NN input)"  )
    
    ## Self-Assembly parameters
    parser.add_argument('--str_index', type=int, default=4, help='Index of target structure on the OVITO particle counter array - def ' \
    'quantify_str -> 0 for FCC, 1 for HCP, 2 for BCC-CsCl, 3 for ICO, 4 for SC-checkerboard, 5 for Cubic Diamond, 6 for Hexagonal Diamond, 7 for Open Honeycomb-Graphene, 8 for Square Single Stripe, 9 for Binary Triangular, 10  for Other')
    parser.add_argument('--ig', type=int, help='Index of goal structure in the goal dictionary')
    parser.add_argument('--three_d', type= int, default= 0, help="IF the goal structure is 3D or 2D so we can ajust the classification")
    parser.add_argument('--goal_dic', type=dict, help='Dictionary containing the goal fractions for each structure')

    ## MD simulation parameters
    parser.add_argument('--system_file', type=str, help = 'File that contains the initial configuration of the system')
    parser.add_argument('--goal_structure_file', type=str, help = 'File that contains the goal structure configuration of the system')
    parser.add_argument('--Nc', type=int, default=int(1e7), help='Number of steps that the simulation will run')
    parser.add_argument('--dt', type= float, default= 1e-3, help="Value of time step in the simulation")
    parser.add_argument('--goal_rdf_json_file', type=str)
    parser.add_argument('--goal_rdf_file', type=str)
    parser.add_argument('--goal', type=float, default=0.8, help="Goal Crystallization fraction")
    parser.add_argument("--goal_vector", type=torch.Tensor, help="One-hot vector indicating the goal structure")
    parser.add_argument('--hour_limit', type = float, default = 10.0, help="Upper Boundary on the MD simulation for debugging or time saving on trivial physical latencies")
    parser.add_argument('--structure_dict', type = dict, help="Dictionary mapping structure indices to names")
    args = parser.parse_args()
  
    # for version in range (2,10):
    args.structure_dict = {
    0: "FCC",          # Face-Centered Cubic
    1: "HCP",          # Hexagonal Close-Packed
    2: "BCC",          # Body-Centered Cubic
    3: "ICO",          # Icosahedral
    4: "SC",           # Simple Cubic or checkerboard
    5: "Cub_Diam",     # Cubic Diamond
    6: "Hex_Diam",     # Hexagonal Diamond
    7: "OHC",          # Possibly Orthorhombic or Other Hexagonal Close variant
    8: "SSS",          # Square single stripe structure
    9: "BTr",          # Binary Triangular
    10: "Other"         # All unidentified or miscellaneous structures
    }   
    if args.three_d == 0:
        goal_dic = {"SC": 0.64, "OHC": 0.72, "SSS": 0.68, "BTr":0.48}
        
    else:
        goal_dic = {"FCC": 0.47, "HCP": 0.9, "BCC": 0.559, "ICO": 0.8, 'SC': 0.646, "Cub_Diam": 0.486, "Hex_Diam": 0.519} #PTM classification
      
    args.goal_dic = goal_dic ; structure_name = args.structure_dict[args.str_index]
    key_list = list(goal_dic.keys()) ; args.ig = key_list.index(structure_name)
    batch_size = args.batch
    epochs = args.epochs
    action_dim = args.action_dim
    args.obs_dim = len(goal_dic)
    args.goal = goal_dic[args.structure_dict[args.str_index]]
    goal_vector = torch.zeros(batch_size, args.obs_dim) ; 
    goal_vector[:,args.ig] = 1.0 ; args.goal_vector = goal_vector
    
    #Results and Model information Directory
    directory_name = "Model_and_Results"
    directory_conf = "Initial_Configurations"
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if not os.path.exists(directory_conf):
        os.makedirs(directory_conf)
    
    # Create organized subdirectory structure
    data_dir = os.path.join(directory_name, "1_data")
    training_dynamics_dir = os.path.join(data_dir, "training_dynamics")
    network_metrics_dir = os.path.join(data_dir, "network_metrics")
    checkpoints_dir = os.path.join(directory_name, "2_checkpoints")
    results_dir = os.path.join(directory_name, "3_results")
    plots_dir = os.path.join(directory_name, "4_plots")
   
    for subdir in [data_dir, training_dynamics_dir, network_metrics_dir,
                   checkpoints_dir, results_dir, plots_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    args.structure = args.structure_dict[args.str_index]
    dir = os.getcwd()
    args.goal_rdf_json_file = f'{dir}/Target_RDF/target_{args.structure}.json'
    if args.three_d == 0:
        args.system_file = os.path.join(directory_conf,"square_lattice_output_equilibriate_minimized.gsd")
        # args.system_file = os.path.join(directory_conf, "initial_conf_a66_b33.gsd")
    if args.three_d == 1:
        args.system_file = os.path.join(directory_conf,"simple_cubic_output_equilibriate_minimized.gsd")
    args.goal_structure_file = os.path.join(directory_conf,f"goal_{args.structure}_structure.gsd")
    
    if not os.path.exists(args.goal_rdf_json_file):
        print(f"[ERROR] File does not exist: {args.goal_rdf_json_file}",flush =True)
        sys.exit(1)
    else:
        print(f"[OK] File found: {args.goal_rdf_json_file}", flush=True)
    first_run = args.first_run

    # Policy/critic networks run on CPU; GPUs are used exclusively by HOOMD simulations
    device = torch.device("cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # 1. Initialize policy and critics
    policy = ActorNet(
        args.obs_dim,
        action_dim,
        batch_size,
        u_min=torch.tensor([0.4, 0.0, 0.0]), # Make sure it is the same with buffer
        u_max=torch.tensor([2.5, 3.0, 3.0])
    ).to(device)
    q1_net = CriticNet(args.obs_dim, args.action_dim).to(device)
    q2_net = CriticNet(args.obs_dim, args.action_dim).to(device)

    # 1.5. Pretrain policy μ to initial conditions (only on first run)
    if first_run == 1 and args.pretrain_mu == 1:
        # Target actions from argparse: [σ_II, λ_II, λ_IJ]
        target_action_single = torch.tensor(args.pretrain_target, dtype=torch.float32)  

        # Repeat for batch: [batch_size, 3]
        target_actions = target_action_single.unsqueeze(0).expand(batch_size, -1).clone()

        # Initial states (random, will learn to produce target for any state)
        initial_states = torch.rand(batch_size, args.obs_dim, dtype=torch.float32) * 0.1

        # Goal vector: use single goal template (will be expanded in pretrain function)
        goal_template = torch.zeros(args.obs_dim)
        goal_template[args.ig] = 1.0

        # Pretrain for 250 steps with enhanced LR (matching REINFORCE)
        policy = pretrain_policy_mean(
            policy, target_actions, initial_states,
            goal_template, device, lr=1.5e-2, num_steps=250
        )

    # 2. Load the state_dict (weights) if resuming
    if first_run==0:
        policy.load_state_dict(torch.load(os.path.join(checkpoints_dir,"policy_model.pt"), map_location=device))
        if os.path.exists(os.path.join(checkpoints_dir,"q1_model.pt")):
            q1_net.load_state_dict(torch.load(os.path.join(checkpoints_dir,"q1_model.pt"), map_location=device))
        if os.path.exists(os.path.join(checkpoints_dir,"q2_model.pt")):
            q2_net.load_state_dict(torch.load(os.path.join(checkpoints_dir,"q2_model.pt"), map_location=device))

    # 3. Set it back to training mode (optional, but good practice)
    policy.train()
    q1_net.train()
    q2_net.train()

    # 4. Recreate the ac_optimizer and optionally load its state_dict (if saved)
    ac_optimizer = optim.Adam(policy.parameters(), lr=args.actor_lr)
    cr1_optimizer = optim.Adam(q1_net.parameters(), lr=args.critic_lr)
    cr2_optimizer = optim.Adam(q2_net.parameters(), lr=args.critic_lr)

    if first_run==0:
        ac_opt_path = os.path.join(checkpoints_dir,"ac_optimizer.pt")
        cr1_opt_path = os.path.join(checkpoints_dir,"cr1_optimizer.pt")
        cr2_opt_path = os.path.join(checkpoints_dir,"cr2_optimizer.pt")
        alpha_opt_path = os.path.join(checkpoints_dir,"alpha_optimizer.pt")
        log_alpha_path = os.path.join(checkpoints_dir,"log_alpha.pt")

        if os.path.exists(ac_opt_path):
            ac_optimizer.load_state_dict(torch.load(ac_opt_path, map_location=device))
        if os.path.exists(cr1_opt_path):
            cr1_optimizer.load_state_dict(torch.load(cr1_opt_path, map_location=device))
        if os.path.exists(cr2_opt_path):
            cr2_optimizer.load_state_dict(torch.load(cr2_opt_path, map_location=device))

    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Batch size:        {batch_size}")
    print(f"Epochs:            {epochs}")
    print(f"Action dim:        {action_dim}")
    print(f"Obs dim:           {args.obs_dim}")
    print(f"Actor LR:          {args.actor_lr}")
    print(f"Critic LR:         {args.critic_lr}")
    print(f"Alpha LR:          {args.alpha_lr}")
    print(f"Target structure:  {args.structure}")
    print(f"{'='*70}\n", flush=True)
    state_history = []
    ac_loss_history = []
    loss_history = []
    state_history = []
    action_log = []
    action_stats_log = []  # Track mean, std, min, max per action dimension per epoch
    best_data_training = []
    reward_log = []
    gradient_history = []
    sigma_log = []  # Track scheduled sigma over epochs
    performance_log = []  # Track performance metric used for noise scheduling

    # NEW TRACKING: Critic loss and Q-value evolution
    critic_loss_log = []  # Track Q1, Q2, combined critic loss per epoch
    q_value_log = []  # Track mean, min, max Q1, Q2 values per epoch

    # NEW TRACKING: Actor distribution parameters
    actor_dist_log = []  # Track μ (mean) and σ (scheduled) statistics per epoch

    # NEW TRACKING: Learning rate schedule
    lr_log = []  # Track actor_lr, critic_lr per epoch

    # NEW TRACKING: Policy convergence metrics
    policy_convergence_log = []  # Track L2 norm of weight changes between epochs
    prev_policy_weights = None  # Store previous epoch's weights for comparison

    # These checkpoints were saved by this codebase and contain numpy arrays.
    # weights_only=False is safe here since the files are from a trusted source.
    def _torch_load(path):
        return torch.load(path, map_location=device, weights_only=False)

    # Load training history if resuming
    if first_run == 0:
        history_path = os.path.join(checkpoints_dir, "training_history.pt")
        if os.path.exists(history_path):
            print(f"[INFO] Loading training history from checkpoint...")
            training_history = _torch_load(history_path)
            action_log = training_history.get('action_log', [])
            action_stats_log = training_history.get('action_stats_log', [])
            reward_log = training_history.get('reward_log', [])
            sigma_log = training_history.get('sigma_log', [])
            performance_log = training_history.get('performance_log', [])
            critic_loss_log = training_history.get('critic_loss_log', [])
            q_value_log = training_history.get('q_value_log', [])
            actor_dist_log = training_history.get('actor_dist_log', [])
            lr_log = training_history.get('lr_log', [])
            policy_convergence_log = training_history.get('policy_convergence_log', [])
            prev_policy_weights = training_history.get('prev_policy_weights', None)
            print(f"[INFO] Loaded {len(reward_log)} epochs of training history")
        else:
            print(f"[WARNING] No saved training history found, starting fresh logs")

    # Track best performing design parameters
    best_performance = -float('inf')  # Track best goal fraction achieved
    best_action_params = None          # Action parameters that achieved best performance
    best_epoch = -1                    # Epoch where best performance occurred

    # Initialize sigma for first epoch (high exploration)
    scale = policy.u_max - policy.u_min
    sigma_current = args.sigma_hi * scale  # Start with maximum exploration

    # Load sigma_current and best tracking if resuming
    if first_run == 0:
        sigma_path = os.path.join(checkpoints_dir, "sigma_current.pt")
        if os.path.exists(sigma_path):
            sigma_current = _torch_load(sigma_path)
            print(f"[INFO] Loaded sigma_current from checkpoint: {sigma_current}")
        else:
            print(f"[WARNING] No saved sigma_current found, starting with sigma_hi")

        best_tracking_path = os.path.join(checkpoints_dir, "best_tracking.pt")
        if os.path.exists(best_tracking_path):
            best_tracking = _torch_load(best_tracking_path)
            best_performance = best_tracking['best_performance']
            best_action_params = best_tracking['best_action_params']
            best_epoch = best_tracking['best_epoch']
            print(f"[INFO] Loaded best tracking: performance={best_performance:.4f} at epoch {best_epoch}")
        else:
            print(f"[WARNING] No saved best tracking found, starting fresh")

    # u_min = torch.tensor([0.5], device=device)
    # u_max = torch.tensor([4.0], device=device)
    if first_run==0:
        # Skip legacy CSV reloads; resume with a small random state
        state = torch.rand(batch_size, args.obs_dim) * 0.1
    else:
        state = torch.rand(batch_size, args.obs_dim) * 0.1
        # print(f"Shape of state: {state.shape}")


    # Buffer setup (HER-style goal conditioning)
    buffer_path = os.path.join(training_dynamics_dir, "buffer.csv")
    # Buffer stores FULL 4D action vector [σ_II, σ_AB, λ_II, λ_IJ]
    buffer_action_dim = 4
    u_min_buf = torch.tensor([0.4, 1.0, 0.0, 0.0])  # Full bounds including fixed σ_AB=1.0
    u_max_buf = torch.tensor([2.5, 1.0, 3.0, 3.0])
    replay_buffer = Buffer(
        buffer_path,
        obs_dim=args.obs_dim,
        action_dim=buffer_action_dim,
        u_min=u_min_buf,
        u_max=u_max_buf,
        goal=args.goal,
        state_bounds=(0.0, 1.0),
    )
    if first_run == 0:
        replay_buffer.load_or_initialize(first_run=0)
        # Get the last epoch from buffer and continue from there
        # Use get_max_epoch if available, otherwise read from buffer data directly
        if hasattr(replay_buffer, 'get_max_epoch'):
            start_epoch = replay_buffer.get_max_epoch() + 1
        else:
            # Fallback: read max epoch directly from buffer data
            if replay_buffer.data:
                df_temp = pd.DataFrame(replay_buffer.data)
                if 'epoch' in df_temp.columns:
                    start_epoch = int(df_temp['epoch'].max()) + 1
                else:
                    start_epoch = 0
            else:
                start_epoch = 0
        print(f"[INFO] Resuming training from epoch {start_epoch} (loaded buffer with max epoch {start_epoch - 1})")
    else:
        replay_buffer.initialize_random(
            num_samples=6,
            batch_size=batch_size,
            calc_state_fn=calc_state,
            parameters=args,
            state_history=state_history,
            start_epoch=-1,  # Pre-training samples labeled as epoch -1
        )
        start_epoch = 0
        print(f"[INFO] Starting fresh training from epoch 0")

    # Print epoch range before starting training
    final_epoch = start_epoch + epochs - 1
    print(f"\n{'='*70}")
    print(f"EPOCH RANGE: {start_epoch} → {final_epoch} ({epochs} epochs total)")
    print(f"{'='*70}\n")
    
    # Training loop: run for 'epochs' iterations starting from start_epoch
    for epoch in range(start_epoch, start_epoch + epochs):

        # Forward pass
        # Ensure shapes are 2D and aligned
        state = state.view(state.shape[0], -1)
        gv = goal_vector
        if gv.dim() == 1:
            gv = gv.unsqueeze(0).expand(state.shape[0], -1)
        sg = torch.cat((state, gv), dim=1)
        print(f"Shape of sg: {sg.shape}")

        # Strategy: Use CURRENT epoch's sigma for rollout, compute NEXT epoch's sigma
        # This gives actor one epoch to shift μ before we adapt noise
        sigma_hi, sigma_lo = args.sigma_hi, args.sigma_lo  # Configurable via argparse
        scale = policy.u_max - policy.u_min

        # Use sigma from previous epoch for THIS epoch's rollout
        sigma = sigma_current

        # Rollout with CURRENT sigma (not next!)
        bound_action, log_prob, mu = policy(sg, sigma, return_dist_params=True)

        # Monitor distribution parameters
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} - POLICY DISTRIBUTION MONITORING")
        print(f"{'='*70}")
        print(f"μ (mean):        {mu.detach().cpu().numpy()}")
        print(f"σ (scheduled):   {sigma.detach().cpu().numpy() if isinstance(sigma, torch.Tensor) else sigma}")
        print(f"\nμ stats:   min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}")
        if isinstance(sigma, torch.Tensor):
            if sigma.numel() == 1:
                print(f"σ (scheduled):  {sigma.item():.4f} (scalar)")
            else:
                print(f"σ stats:   min={sigma.min().item():.4f}, max={sigma.max().item():.4f}, mean={sigma.mean().item():.4f}")
        else:
            print(f"σ (scheduled):  {sigma:.4f} (scalar)")
        print(f"{'='*70}\n")

        # Detach reward from computation graph
        print("Calculating next state and reward... for epoch:", epoch)
        with torch.no_grad():
            next_state, reward = calc_state(state,bound_action,batch_size, args, state_history, epoch)
        
        # Store transitions in replay buffer, recording best structure index
        next_state_np = next_state.detach().cpu().numpy()
        reward_np = reward.detach().cpu().numpy()
        state_np = state.detach().cpu().numpy()
        action_np = bound_action.detach().cpu().numpy()

        # Early stopping check: Monitor performance on target goal
        goal_fractions = next_state_np[:, args.ig]  # Extract target structure fraction for all samples
        avg_performance = goal_fractions.mean()
        batch_best_performance = goal_fractions.max()

        # Track best performing action parameters across all epochs
        if batch_best_performance > best_performance:
            best_performance = batch_best_performance
            best_idx_in_batch = goal_fractions.argmax()
            best_action_params = action_np[best_idx_in_batch].copy()
            best_epoch = epoch
            print(f"  New best performance! {best_performance:.4f} at epoch {epoch}")

        print(f"Epoch {epoch} - Target Goal ({args.structure}) Performance:")
        print(f"  Average: {avg_performance:.4f} | Best in batch: {batch_best_performance:.4f}")
        print(f"  Global best: {best_performance:.4f} (epoch {best_epoch})")

        # Compute rollout-based successes for noise scheduler and early stopping
        num_successes = (goal_fractions >= args.success_threshold).sum()  # Count based on configurable threshold
        current_performance = num_successes / batch_size  # Range: [0, 1]

        # Compute NEXT epoch's sigma based on rollout success rate
        if epoch < args.exploration_epochs:
            next_noise_coeff = sigma_hi  # High exploration during exploration phase
        else:
            next_noise_coeff = (sigma_hi - sigma_lo) * (1.0 - current_performance)**float(batch_size) + sigma_lo
        sigma_next = next_noise_coeff * scale

        print(f"Noise scheduler: successes={num_successes}/{batch_size}, perf={current_performance:.4f}, σ_current={sigma[0]:.4f}, σ_next={sigma_next[0]:.4f}")
        # HYBRID HER: Store each transition TWICE with different goals
        for i in range(batch_size):
            ns_i = next_state_np[i]
            s_i = state_np[i]
            a_i = action_np[i]
            r_original = reward_np[i]  # Already computed for original goal

            # FIRST ADDITION: Store with ORIGINAL goal
            replay_buffer.add(
                s_i,
                a_i,
                r_original,  # Reward for original goal
                ns_i,
                best_idx=args.ig,  # Original goal index
                goal=args.ig,      # Original goal
                epoch=epoch,  # Use actual epoch number (no offset)
            )

            # SECOND ADDITION: Store with ACHIEVED goal (HER)
            achieved_idx = int(np.argmax(ns_i)) if ns_i.size > 1 else args.ig

            # Only add HER transition if achieved goal differs from original
            if achieved_idx != args.ig:
                # RECOMPUTE reward for achieved goal
                r_achieved = compute_goal_reward(ns_i, achieved_idx, success_threshold=args.success_threshold)

                replay_buffer.add(
                    s_i,
                    a_i,
                    r_achieved,  # Reward for achieved goal CORRECT!
                    ns_i,
                    best_idx=achieved_idx,  # Achieved goal index
                    goal=achieved_idx,      # Achieved goal
                    epoch=epoch,  # Use actual epoch number (no offset)
                )
        # Stop training if criteria met: 100% success rate in rollout
        if num_successes == batch_size:
            print("\n" + "="*70)
            print("EARLY STOPPING TRIGGERED!")
            print(f"   Target structure ({args.structure}) achieved:")
            print(f"   Average performance: {avg_performance:.4f} (> 0.8)")
            print(f"   Best in batch: {batch_best_performance:.4f} (> 0.9)")
            print(f"   Training stopped at epoch {epoch}/{start_epoch + epochs - 1}")
            print("="*70 + "\n")

            # Save final models before stopping
            replay_buffer.save()
            torch.save(policy.state_dict(), os.path.join(checkpoints_dir, "policy_model.pt"))
            torch.save(q1_net.state_dict(), os.path.join(checkpoints_dir, "q1_model.pt"))
            torch.save(q2_net.state_dict(), os.path.join(checkpoints_dir, "q2_model.pt"))
            torch.save(ac_optimizer.state_dict(), os.path.join(checkpoints_dir, "ac_optimizer.pt"))
            torch.save(cr1_optimizer.state_dict(), os.path.join(checkpoints_dir, "cr1_optimizer.pt"))
            torch.save(cr2_optimizer.state_dict(), os.path.join(checkpoints_dir, "cr2_optimizer.pt"))

            # Save sigma_current and best tracking for early stopping
            torch.save(sigma_current, os.path.join(checkpoints_dir, "sigma_current.pt"))
            best_tracking = {
                'best_performance': best_performance,
                'best_action_params': best_action_params,
                'best_epoch': best_epoch
            }
            torch.save(best_tracking, os.path.join(checkpoints_dir, "best_tracking.pt"))

            # Save training history logs
            training_history = {
                'action_log': action_log,
                'action_stats_log': action_stats_log,
                'reward_log': reward_log,
                'sigma_log': sigma_log,
                'performance_log': performance_log,
                'critic_loss_log': critic_loss_log,
                'q_value_log': q_value_log,
                'actor_dist_log': actor_dist_log,
                'lr_log': lr_log,
                'policy_convergence_log': policy_convergence_log,
                'prev_policy_weights': prev_policy_weights
            }
            torch.save(training_history, os.path.join(checkpoints_dir, "training_history.pt"))

            # Save early stopping info
            with open(os.path.join(results_dir, "early_stopping_info.txt"), "w") as f:
                f.write(f"Early stopping triggered at epoch {epoch}/{start_epoch + epochs - 1}\n")
                f.write(f"Started from epoch: {start_epoch}\n")
                f.write(f"Target structure: {args.structure}\n")
                f.write(f"Average performance: {avg_performance:.6f}\n")
                f.write(f"Best in batch: {batch_best_performance:.6f}\n")
                f.write(f"Global best performance: {best_performance:.6f} (epoch {best_epoch})\n")

            break  # Exit training loop
        # ========================================================================
        # NETWORK TRAINING: 10 critic updates, 1 actor update
        # Strategy: Train critic thoroughly to learn Q-landscape, update actor sparingly
        # to avoid drift from pretrained good region when rewards are sparse
        # ========================================================================
        num_critic_steps = 10
        num_actor_steps = 1

        for gradient_step in range(num_critic_steps):
            # Sample fresh batch from buffer for each gradient step
            # After exploration phase, sample ONLY target goal transitions
            if epoch >= args.exploration_epochs:
                dict_buffer = replay_buffer.sample_target_goal(batch_size, target_goal=args.ig, strategy='meaningful_bias')
            else:
                # During exploration, sample from all goals
                dict_buffer = replay_buffer.sample(batch_size, strategy='meaningful_bias')

            # Convert to leaf tensors (requires_grad=False by default)
            s = torch.as_tensor(dict_buffer['states'], dtype=torch.float32)
            a_full = torch.as_tensor(dict_buffer['actions'], dtype=torch.float32)  # 4D: [σ_II, σ_AB, λ_II, λ_IJ]
            a = a_full[:, [0, 2, 3]]  # Extract only learned dims: [σ_II, λ_II, λ_IJ] for critic
            reward_sample = torch.as_tensor(dict_buffer['rewards'], dtype=torch.float32)

            # ========== LR BOOST DETECTION (DISABLED) ==========
            # Removed LR boost for baseline performance testing
            # Track successes for monitoring only
            has_success_in_batch = (reward_sample == 0.0).any()
            num_successes = (reward_sample == 0.0).sum().item()
            if has_success_in_batch:
                print(f"  [Step {gradient_step+1}/{num_critic_steps}] Training on {num_successes} SUCCESS(es) (no LR boost)")

            lr_multiplier = 1.0  # Always 1.0 (no boost)
            best_idx_sample = dict_buffer.get("best_idx")
            if best_idx_sample is None:
                best_idx_sample = dict_buffer.get("goals")
            g_vec = torch.zeros((s.shape[0], args.obs_dim), dtype=torch.float32)
            g_idx = torch.as_tensor(best_idx_sample, dtype=torch.long)
            valid_mask = (g_idx >= 0) & (g_idx < args.obs_dim)
            g_vec[torch.arange(s.shape[0])[valid_mask], g_idx[valid_mask]] = 1.0
            g_scalar = g_idx.float().unsqueeze(1)
            sag = torch.cat((s, a, g_vec), dim=1)
            q1_value = q1_net(sag).view(-1)  # [batch_size]
            q2_value = q2_net(sag).view(-1)  # [batch_size]

            # ========== CRITIC UPDATE ==========
            critic_loss_fn = nn.MSELoss()
            target_q = reward_sample.view(-1).detach()
            critic_loss = critic_loss_fn(q1_value, target_q) + critic_loss_fn(q2_value, target_q)

            cr1_optimizer.zero_grad()
            cr2_optimizer.zero_grad()
            critic_loss.backward()
            cr1_optimizer.step()
            cr2_optimizer.step()

            # ========== ACTOR UPDATE (only on last critic step) ==========
            # Strategy: Update actor sparingly to avoid drift from pretrained region
            # Only update once per epoch after critic has learned the landscape
            if gradient_step == num_critic_steps - 1:
                # Sample actions from current policy
                pi_action, log_prob_pi = policy(torch.cat((s, g_vec), dim=1), sigma)
                pi_action_learned = pi_action[:, [0, 2, 3]]
                sag_pi = torch.cat((s, pi_action_learned, g_vec), dim=1)
                q1_pi = q1_net(sag_pi).squeeze()
                q2_pi = q2_net(sag_pi).squeeze()
                q_min_pi = torch.min(q1_pi, q2_pi)

                # Actor loss: Simply maximize Q (no entropy term)
                # μ learns "where are good actions?" σ handles "how much to explore"
                a = alpha_scheduler(epoch)
                print("Temperature alpha is:", a)
                actor_loss = (a*log_prob_pi - q_min_pi).mean()

                # Standard update (no LR boost, no gradient clipping)
                ac_optimizer.zero_grad()
                actor_loss.backward()
                ac_optimizer.step()

                # Collect gradient norms for logging
                gradient_row = []
                for name, param in policy.named_parameters():
                    if param.grad is not None:
                        gradient_row.append(param.grad.norm().view(-1))
                gradient_row = torch.cat(gradient_row).cpu().numpy()
                gradient_history.append(gradient_row)

            # Print diagnostics only on last gradient step
            if gradient_step == num_critic_steps - 1:
                # ========== LOSS MONITORING ==========
                print(f"\n{'='*70}")
                print(f"LOSS MONITORING (Epoch {epoch}, Step {gradient_step+1}/{num_critic_steps})")
                print(f"{'='*70}")
                print(f"Critic Loss (total):       {critic_loss.item():.6f}")
                print(f"  └─ Q1 MSE:               {critic_loss_fn(q1_value, target_q).item():.6f}")
                print(f"  └─ Q2 MSE:               {critic_loss_fn(q2_value, target_q).item():.6f}")
                print(f"  └─ Q1 mean:              {q1_value.mean().item():.6f}")
                print(f"  └─ Q2 mean:              {q2_value.mean().item():.6f}")
                print(f"  └─ Target (reward) mean: {target_q.mean().item():.6f}")
                print(f"\nActor Loss:                {actor_loss.item():.6f}")
                print(f"  └─ Q-value term:         {-q_min_pi.mean().item():.4f}")
                print(f"  └─ Q_min mean:           {q_min_pi.mean().item():.6f}")
                print(f"{'='*70}\n")

                # ========== GRADIENT DIAGNOSTICS ==========
                print(f"\n{'='*70}")
                # print(f"GRADIENT DIAGNOSTICS (Epoch {epoch}, Step {gradient_step+1}/{num_gradient_steps})")
                print(f"{'='*70}")

                print(f"\n[ACTOR GRADIENTS]")
                for name, param in policy.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_min = param.grad.min().item()
                        grad_max = param.grad.max().item()
                        grad_mean = param.grad.mean().item()
                        print(f"{name:30s} | norm: {grad_norm:10.6f} | min: {grad_min:10.6f} | max: {grad_max:10.6f} | mean: {grad_mean:10.6f}")

                        if "net.6" in name:
                            print(f"  WARNING: LAST LAYER (outputs μ and log_std)")
                    else:
                        print(f"{name:30s} | grad is None [MISSING]")

                print(f"\n[CRITIC Q1 GRADIENTS]")
                for name, param in q1_net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"{name:30s} | norm: {grad_norm:10.6f}")
                    else:
                        print(f"{name:30s} | grad is None [MISSING]")

                print(f"\n[CRITIC Q2 GRADIENTS]")
                for name, param in q2_net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"{name:30s} | norm: {grad_norm:10.6f}")
                    else:
                        print(f"{name:30s} | grad is None [MISSING]")

                print(f"{'='*70}\n")

        # End of gradient step loop - now do per-epoch logging
        state = next_state.view(batch_size, -1)

        # Logging (always detach before storing)
        loss_history.append(actor_loss.item())
        state_history.append(state.detach().cpu().numpy().mean())
        bound_action_mean = bound_action.mean(dim=0).detach().cpu()  # Detach before storing
        action_log.append(bound_action_mean.cpu().numpy())

        # NEW: Track action statistics (mean, std, min, max) per dimension
        bound_action_np = bound_action.detach().cpu().numpy()
        action_stats = {
            'mean': bound_action_np.mean(axis=0),
            'std': bound_action_np.std(axis=0),
            'min': bound_action_np.min(axis=0),
            'max': bound_action_np.max(axis=0)
        }
        action_stats_log.append(action_stats)

        # Track average reward for the ORIGINAL GOAL (the one we're training for)
        reward_mean = reward.detach().cpu().numpy().mean()
        reward_log.append(reward_mean)

        # Track sigma and performance for noise scheduler validation
        if isinstance(sigma, torch.Tensor):
            sigma_mean = sigma.mean().item() if sigma.numel() > 1 else sigma.item()
        else:
            sigma_mean = sigma
        sigma_log.append(sigma_mean)
        performance_log.append(current_performance)  # Log current performance

        # NEW: Track critic loss and Q-value evolution (from last gradient step)
        q1_loss_val = critic_loss_fn(q1_value, target_q).item()
        q2_loss_val = critic_loss_fn(q2_value, target_q).item()
        critic_loss_log.append({
            'q1_loss': q1_loss_val,
            'q2_loss': q2_loss_val,
            'combined_loss': critic_loss.item()
        })

        q_value_log.append({
            'q1_mean': q1_value.mean().item(),
            'q1_min': q1_value.min().item(),
            'q1_max': q1_value.max().item(),
            'q2_mean': q2_value.mean().item(),
            'q2_min': q2_value.min().item(),
            'q2_max': q2_value.max().item()
        })

        # NEW: Track actor distribution parameters (μ and σ) from rollout
        mu_np = mu.detach().cpu().numpy()
        actor_dist_log.append({
            'mu_mean': mu_np.mean(axis=0),  # Mean of μ across batch, per action dim
            'mu_std': mu_np.std(axis=0),    # Std of μ across batch, per action dim
            'mu_min': mu_np.min(axis=0),    # Min of μ across batch, per action dim
            'mu_max': mu_np.max(axis=0),    # Max of μ across batch, per action dim
            'sigma_scheduled': sigma.detach().cpu().numpy() if isinstance(sigma, torch.Tensor) else np.array([sigma] * action_dim)
        })

        # NEW: Track learning rates
        current_actor_lr = ac_optimizer.param_groups[0]['lr']
        current_critic_lr = cr1_optimizer.param_groups[0]['lr']
        lr_log.append({
            'actor_lr': current_actor_lr,
            'critic_lr': current_critic_lr,
            'alpha_lr': args.alpha_lr  # Static, but tracked for completeness
        })

        # NEW: Track policy convergence (L2 norm of weight changes)
        current_policy_weights = {name: param.clone().detach().cpu() for name, param in policy.named_parameters()}

        if prev_policy_weights is not None:
            # Compute L2 norm of weight differences: ||θ_t - θ_{t-1}||
            weight_diff_norm = 0.0
            for name in current_policy_weights:
                diff = current_policy_weights[name] - prev_policy_weights[name]
                weight_diff_norm += torch.sum(diff ** 2).item()
            weight_diff_norm = np.sqrt(weight_diff_norm)

            policy_convergence_log.append({
                'param_change_l2': weight_diff_norm
            })
        else:
            # First epoch: no previous weights to compare
            policy_convergence_log.append({
                'param_change_l2': np.nan  # NaN for first epoch
            })

        # Store current weights for next epoch comparison
        prev_policy_weights = current_policy_weights

        # Update sigma for NEXT epoch
        sigma_current = sigma_next

        # Detach next_state before using in next epoch
        

        # # # Save model
        replay_buffer.save()
        torch.save(policy.state_dict(), os.path.join(checkpoints_dir, "policy_model.pt"))
        torch.save(q1_net.state_dict(), os.path.join(checkpoints_dir, "q1_model.pt"))
        torch.save(q2_net.state_dict(), os.path.join(checkpoints_dir, "q2_model.pt"))
        torch.save(ac_optimizer.state_dict(), os.path.join(checkpoints_dir, "ac_optimizer.pt"))
        torch.save(cr1_optimizer.state_dict(), os.path.join(checkpoints_dir, "cr1_optimizer.pt"))
        torch.save(cr2_optimizer.state_dict(), os.path.join(checkpoints_dir, "cr2_optimizer.pt"))

        # Save sigma_current for resume
        torch.save(sigma_current, os.path.join(checkpoints_dir, "sigma_current.pt"))

        # Save best performance tracking
        best_tracking = {
            'best_performance': best_performance,
            'best_action_params': best_action_params,
            'best_epoch': best_epoch
        }
        torch.save(best_tracking, os.path.join(checkpoints_dir, "best_tracking.pt"))

        # Save training history logs for resume
        training_history = {
            'action_log': action_log,
            'action_stats_log': action_stats_log,
            'reward_log': reward_log,
            'sigma_log': sigma_log,
            'performance_log': performance_log,
            'critic_loss_log': critic_loss_log,
            'q_value_log': q_value_log,
            'actor_dist_log': actor_dist_log,
            'lr_log': lr_log,
            'policy_convergence_log': policy_convergence_log,
            'prev_policy_weights': prev_policy_weights
        }
        torch.save(training_history, os.path.join(checkpoints_dir, "training_history.pt"))

        if gradient_history:
            gradients = np.vstack(gradient_history)
            np.savetxt(os.path.join(network_metrics_dir, "gradients.csv"), gradients, delimiter=",", header="Norms per layer for weigh and bias gradients", comments="")

    # Save training logs to CSV for plotting (AFTER training loop completes)
    action_array = np.vstack(action_log)
    reward_array = np.array(reward_log)

    # Create epochs array: always start from 0 (includes loaded history + new epochs)
    epochs_array = np.arange(len(reward_log))

    # Save actions log with statistics (mean, std, min, max) per action dimension
    action_names = ['sigma_II', 'sigma_AB', 'lambda_II', 'lambda_AB']
    action_data = {'epoch': epochs_array}

    for i, name in enumerate(action_names):
        action_data[f'{name}_mean'] = action_array[:, i]
        action_data[f'{name}_std'] = np.array([stats['std'][i] for stats in action_stats_log])
        action_data[f'{name}_min'] = np.array([stats['min'][i] for stats in action_stats_log])
        action_data[f'{name}_max'] = np.array([stats['max'][i] for stats in action_stats_log])

    action_df = pd.DataFrame(action_data)
    action_df.to_csv(os.path.join(training_dynamics_dir, "actions_per_epoch.csv"), index=False)
    print(f"Saved action statistics to {training_dynamics_dir}/actions_per_epoch.csv")

    # Save rewards log (average reward for original goal per epoch)
    reward_df = pd.DataFrame({"epoch": epochs_array, "avg_reward": reward_array})
    reward_df.to_csv(os.path.join(training_dynamics_dir, "rewards_per_epoch.csv"), index=False)
    print(f"Saved rewards to {training_dynamics_dir}/rewards_per_epoch.csv")

    # Save sigma and performance logs for noise scheduler validation
    sigma_array = np.array(sigma_log)
    performance_array = np.array(performance_log)
    scheduler_df = pd.DataFrame({
        "epoch": epochs_array,
        "sigma": sigma_array,
        "success_rate": performance_array,  # Now based on rollout successes, not Q-values
        "noise_coeff": sigma_array / (policy.u_max - policy.u_min).numpy().mean()  # Recover coefficient
    })
    scheduler_df.to_csv(os.path.join(training_dynamics_dir, "noise_scheduler_per_epoch.csv"), index=False)
    print(f"Saved noise scheduler validation data to {training_dynamics_dir}/noise_scheduler_per_epoch.csv")

    # NEW: Save critic loss per epoch
    critic_loss_df = pd.DataFrame({
        'epoch': epochs_array,
        'q1_loss': [log['q1_loss'] for log in critic_loss_log],
        'q2_loss': [log['q2_loss'] for log in critic_loss_log],
        'combined_loss': [log['combined_loss'] for log in critic_loss_log]
    })
    critic_loss_df.to_csv(os.path.join(network_metrics_dir, "critic_loss_per_epoch.csv"), index=False)
    print(f"Saved critic loss data to {network_metrics_dir}/critic_loss_per_epoch.csv")

    # NEW: Save Q-value evolution per epoch
    q_value_df = pd.DataFrame({
        'epoch': epochs_array,
        'q1_mean': [log['q1_mean'] for log in q_value_log],
        'q1_min': [log['q1_min'] for log in q_value_log],
        'q1_max': [log['q1_max'] for log in q_value_log],
        'q2_mean': [log['q2_mean'] for log in q_value_log],
        'q2_min': [log['q2_min'] for log in q_value_log],
        'q2_max': [log['q2_max'] for log in q_value_log]
    })
    q_value_df.to_csv(os.path.join(network_metrics_dir, "q_values_per_epoch.csv"), index=False)
    print(f"Saved Q-value evolution data to {network_metrics_dir}/q_values_per_epoch.csv")

    # NEW: Save actor distribution parameters per epoch
    actor_dist_data = {'epoch': epochs_array}
    action_dim_names = ['sigma_II', 'lambda_AA', 'lambda_AB']  # 4D learned action space (σ_AB is fixed)

    for i, name in enumerate(action_dim_names):
        actor_dist_data[f'{name}_mu_mean'] = np.array([log['mu_mean'][i] for log in actor_dist_log])
        actor_dist_data[f'{name}_mu_std'] = np.array([log['mu_std'][i] for log in actor_dist_log])
        actor_dist_data[f'{name}_mu_min'] = np.array([log['mu_min'][i] for log in actor_dist_log])
        actor_dist_data[f'{name}_mu_max'] = np.array([log['mu_max'][i] for log in actor_dist_log])

    # Add scheduled sigma (same for all action dims initially, but track separately)
    for i, name in enumerate(action_dim_names):
        actor_dist_data[f'{name}_sigma_scheduled'] = np.array([log['sigma_scheduled'][i] if hasattr(log['sigma_scheduled'], '__len__') else log['sigma_scheduled'] for log in actor_dist_log])

    actor_dist_df = pd.DataFrame(actor_dist_data)
    actor_dist_df.to_csv(os.path.join(network_metrics_dir, "actor_distribution_per_epoch.csv"), index=False)
    print(f"Saved actor distribution parameters to {network_metrics_dir}/actor_distribution_per_epoch.csv")

    # NEW: Save learning rates per epoch
    lr_df = pd.DataFrame({
        'epoch': epochs_array,
        'actor_lr': [log['actor_lr'] for log in lr_log],
        'critic_lr': [log['critic_lr'] for log in lr_log],
        'alpha_lr': [log['alpha_lr'] for log in lr_log]
    })
    lr_df.to_csv(os.path.join(network_metrics_dir, "learning_rates_per_epoch.csv"), index=False)
    print(f"Saved learning rate schedule to {network_metrics_dir}/learning_rates_per_epoch.csv")

    # NEW: Save policy convergence metrics per epoch
    convergence_df = pd.DataFrame({
        'epoch': epochs_array,
        'param_change_l2': [log['param_change_l2'] for log in policy_convergence_log]
    })
    convergence_df.to_csv(os.path.join(network_metrics_dir, "policy_convergence_per_epoch.csv"), index=False)
    print(f"Saved policy convergence metrics to {network_metrics_dir}/policy_convergence_per_epoch.csv")

    # Save best performing design parameters to CSV
    if best_action_params is not None:
        best_params_df = pd.DataFrame({
            'Parameter': ['σ_II (Attractive)', 'σ_AB (Fixed)', 'λ_II (Range)', 'λ_AB (Range)'],
            'Value': best_action_params * 4,
            'Performance': [best_performance] * 4 ,
            'Epoch': [best_epoch] * 4 
        })
        best_params_df.to_csv(os.path.join(results_dir, "best_design_parameters.csv"), index=False)
        print(f"Saved best parameters to {results_dir}/best_design_parameters.csv")
        print(f"\nBEST PERFORMANCE: {best_performance:.4f} at epoch {best_epoch}")
        print(f"   Best action parameters: {best_action_params}")
    else:
        print("\nWARNING: No best parameters recorded (training may have stopped early)")

    # Visualize results of this run
    try:
        from Make_figures import build_run_config, create_figure
        fig_config = build_run_config(args.str_index, args.three_d)
        create_figure(directory_name, fig_config, plots_dir)
        print(f"[INFO] Figures saved to {plots_dir}")
    except Exception as e:
        print(f"[WARNING] Figure generation failed: {e}")

    # Print summary of saved files
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FILE ORGANIZATION SUMMARY")
    print("="*80)
    print(f"\nAll results saved in: {directory_name}/")
    print("\n├── 1_data/")
    print("│   ├── training_dynamics/")
    print("│   │   ├── actions_per_epoch.csv          (Action stats: mean, std, min, max)")
    print("│   │   ├── rewards_per_epoch.csv          (Reward evolution)")
    print("│   │   ├── noise_scheduler_per_epoch.csv  (Sigma and success rate)")
    print("│   │   └── buffer.csv                     (Replay buffer transitions)")
    print("│   └── network_metrics/")
    print("│       ├── critic_loss_per_epoch.csv      (Q1/Q2 loss evolution)")
    print("│       ├── q_values_per_epoch.csv         (Q-value statistics)")
    print("│       ├── actor_distribution_per_epoch.csv (Policy μ and σ)")
    print("│       ├── learning_rates_per_epoch.csv   (LR schedule)")
    print("│       ├── policy_convergence_per_epoch.csv (Weight change L2 norm)")
    print("│       └── gradients.csv                  (Gradient norms per layer)")
    print("├── 2_checkpoints/")
    print("│   ├── policy_model.pt                   (Actor network weights)")
    print("│   ├── q1_model.pt, q2_model.pt         (Critic network weights)")
    print("│   ├── training_history.pt               (All training logs)")
    print("│   ├── sigma_current.pt                  (Current sigma value)")
    print("│   ├── best_tracking.pt                  (Best performance tracking)")
    print("│   └── *_optimizer.pt                    (Optimizer states)")
    print("├── 3_results/")
    print("│   ├── best_design_parameters.csv         (Optimal parameters found)")
    print("│   └── early_stopping_info.txt            (If early stopping triggered)")
    print("└── 4_plots/")
    print("="*80 + "\n")