import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from MD_engine import calc_state

class Buffer:
    """
    Simple experience buffer that can load/save transitions to CSV.
    Each row stores: s, a, r, s', g.
    """

    def __init__(
        self,
        buffer_path,
        obs_dim,
        action_dim,
        u_min,
        u_max,
        goal=0.0,
        state_bounds=(0.0, 1.0),
    ):
        self.buffer_path = buffer_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal = goal
        self.state_bounds = state_bounds

        self.u_min = self._to_np(u_min, action_dim)
        self.u_max = self._to_np(u_max, action_dim)

        self.state_cols = [f"state_{i}" for i in range(obs_dim)]
        self.action_cols = [f"action_{i}" for i in range(action_dim)]
        self.next_state_cols = [f"next_state_{i}" for i in range(obs_dim)]
        self.reward_col = "reward"
        self.goal_col = "goal"
        self.best_idx_col = "best_index"
        self.epoch_col = "epoch"

        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
        self.data = []

    def _to_np(self, arr_like, dim):
        arr = arr_like.detach().cpu().numpy() if isinstance(arr_like, torch.Tensor) else np.array(arr_like)
        arr = np.asarray(arr, dtype=np.float32).flatten()
        if arr.size == 1:
            arr = np.repeat(arr, dim)
        return arr

    def add(self, state, action, reward, next_state, best_idx=None, goal=None, epoch=0):
        """
        Add a full batch of transitions.
        Accepts single or batched inputs. If best_idx is not provided,
        it is inferred via argmax over next_state[i].
        Epoch is recorded per-sample (default 0 for pretraining).
        """
        # Convert to numpy arrays (guaranteed leaf data)
        state_np = np.asarray(state, dtype=np.float32)          # shape: [B, obs_dim] or [obs_dim]
        action_np = np.asarray(action, dtype=np.float32)        # shape: [B, act_dim] or [act_dim]
        reward_np = np.asarray(reward, dtype=np.float32).reshape(-1)  # shape: [B]
        next_state_np = np.asarray(next_state, dtype=np.float32) # shape: [B, obs_dim] or [obs_dim]

        # Ensure batch dimension
        if state_np.ndim == 1:
            state_np = state_np[None, :]
            action_np = action_np[None, :]
            reward_np = reward_np[None]
            next_state_np = next_state_np[None, :]

        batch_size = state_np.shape[0]

        for i in range(batch_size):
            s_i = state_np[i]           # (obs_dim,)
            a_i = action_np[i]          # (act_dim,)
            r_i = reward_np[i]          # scalar
            ns_i = next_state_np[i]     # (obs_dim,)

            best_idx_out = int(best_idx) if best_idx is not None else int(np.argmax(ns_i))
            goal_val = int(goal) if goal is not None else best_idx_out

            row = {}
            for col_name, col_val in zip(self.state_cols, s_i):
                row[col_name] = float(col_val)
            for col_name, col_val in zip(self.action_cols, a_i):
                row[col_name] = float(col_val)
            for col_name, col_val in zip(self.next_state_cols, ns_i):
                row[col_name] = float(col_val)

            row[self.reward_col] = float(r_i)
            row[self.goal_col] = goal_val
            row[self.best_idx_col] = best_idx_out
            row[self.epoch_col] = int(epoch)

            self.data.append(row)

    def save(self):
        if not self.data:
            print(f"⚠️  WARNING: Buffer.save() called but buffer is EMPTY! No data to save.")
            print(f"   Buffer has {len(self.data)} transitions")
            return
        print(f"✓ Saving buffer with {len(self.data)} transitions to {self.buffer_path}")
        df = pd.DataFrame(self.data)

        # Define column order for better readability: epoch, goal info, state, action, reward, next_state
        ordered_cols = [self.epoch_col, self.goal_col, self.best_idx_col]
        ordered_cols += self.state_cols
        ordered_cols += self.action_cols
        ordered_cols += [self.reward_col]
        ordered_cols += self.next_state_cols

        # Reorder columns (keep only existing ones)
        existing_cols = [col for col in ordered_cols if col in df.columns]
        df = df[existing_cols]

        # Format numeric columns consistently for better alignment
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in [self.epoch_col, self.goal_col, self.best_idx_col]:
                # Integer columns
                df[col] = df[col].astype(int)
            else:
                # Float columns: consistent decimal places
                df[col] = df[col].apply(lambda x: f"{x:.6f}" if not pd.isna(x) else "")

        df.to_csv(self.buffer_path, index=False)
        print(f"  OK:  Successfully saved buffer to: {self.buffer_path}")

    def load(self):
        if not os.path.exists(self.buffer_path):
            raise FileNotFoundError(f"No buffer file found at {self.buffer_path}")
        df = pd.read_csv(self.buffer_path)
        self.data = df.to_dict(orient="records")
        return df

    def initialize_random(
        self,
        num_samples=6,
        batch_size=6,
        calc_state_fn=calc_state,
        parameters=None,
        state_history=None,
        start_epoch=0,
    ):
        """
        Seed the buffer with random samples and evaluated rewards.

        Args:
            num_samples (int): Total samples to create.
            batch_size (int): Batch size per calc_state call.
            calc_state_fn (callable): Function(state, actions, batch_size, parameters, state_history, epoch) -> (next_state, reward).
            parameters: Passed through to calc_state_fn.
            state_history: Passed through to calc_state_fn (default empty list).
            start_epoch (int): Epoch counter for logging purposes.
        """
        print(f"\n{'='*60}")
        print(f"INITIALIZING BUFFER with {num_samples} random samples")
        print(f"Batch size: {batch_size}, Action dim: {self.action_dim}")
        print(f"{'='*60}\n")

        if calc_state_fn is None:
            raise ValueError("calc_state_fn must be provided to evaluate rewards.")

        low_s, high_s = self.state_bounds
        remaining = num_samples
        epoch = start_epoch
        state_history = state_history or []

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            states = np.random.uniform(low_s, high_s, size=(current_batch, self.obs_dim)).astype(np.float32)
            # Generate random 4D actions [σ_II, σ_AB, λ_II, λ_IJ] with σ_AB fixed at 1.0
            actions = np.random.uniform(self.u_min, self.u_max, size=(current_batch, self.action_dim)).astype(np.float32)

            state_t = torch.tensor(states, dtype=torch.float32)
            action_t = torch.tensor(actions, dtype=torch.float32)  # Already 4D, no reconstruction needed
            next_state_t, reward_t = calc_state_fn(state_t, action_t, current_batch, parameters, state_history, epoch)

            next_states = next_state_t.detach().cpu().numpy()
            rewards = reward_t.detach().cpu().numpy()

            self.add(state_t, actions, rewards, next_states, epoch=0)

            remaining -= current_batch
            epoch += 1

        self.save()

    def load_or_initialize(self, first_run, num_samples=6):
        num_samples = self.batch_size if hasattr(self, 'batch_size') else num_samples
        if first_run == 0 and os.path.exists(self.buffer_path):
            return self.load()
        self.initialize_random(num_samples)
        return pd.DataFrame(self.data)
    def conc(self, state, goal, action = None):
        if action is None:
            sg = torch.cat((state, goal), dim=1)
            return sg
        else:
            sa = torch.cat((state, action), dim=1)
            sag = torch.cat((sa, action), dim=1)
            return sag
        
    def sample(self, n, strategy='meaningful_bias'):
        """
        Sample n transitions from buffer with configurable strategy.

        Args:
            n: number of samples
            strategy: 'uniform', 'reward_squared', or 'meaningful_bias'
        """
        if not self.data:
            raise ValueError("Buffer is empty; cannot sample.")

        df = pd.DataFrame(self.data)
        rewards = df[self.reward_col].to_numpy(dtype=np.float32)

        if strategy == 'uniform':
            # Uniform sampling: each transition has equal probability
            idxs = np.random.choice(len(df), size=n, replace=True)

        elif strategy == 'reward_squared':
            # Original behavior: weight by reward²
            weights = np.square(rewards)
            if np.all(weights == 0):
                weights = np.ones_like(weights, dtype=np.float32)
            probs = weights / weights.sum()
            idxs = np.random.choice(len(df), size=n, p=probs, replace=True)

        elif strategy == 'meaningful_bias':
            # NEW STRATEGY: 1 random success (if exists) + uniform sampling
            # Prevents Q-overfitting while guaranteeing success signal for sparse rewards
            # Rationale: With sparse rewards, we need success examples but must avoid
            # training the critic on 88% duplicate successes which causes Q-collapse

            success_mask = (rewards == 0.0)
            success_idxs = np.where(success_mask)[0]

            if len(success_idxs) > 0:
                # Sample 1 random success transition
                forced_success_idx = np.random.choice(success_idxs, size=1)

                # Sample remaining (n-1) uniformly from entire buffer
                uniform_idxs = np.random.choice(len(df), size=n-1, replace=True)

                # Combine: [1 success, n-1 uniform samples]
                idxs = np.concatenate([forced_success_idx, uniform_idxs])

                print(f" GOAL Buffer: 1 success + {n-1} uniform samples (from {len(success_idxs)} successes available)")
            else:
                # No successes yet - pure uniform sampling
                idxs = np.random.choice(len(df), size=n, replace=True)
                print(f" WARNING Buffer: No successes yet - pure uniform sampling ({n} samples)")

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Extract sampled data
        sample_df = df.iloc[idxs]

        states = sample_df[self.state_cols].to_numpy(dtype=np.float32)
        actions = sample_df[self.action_cols].to_numpy(dtype=np.float32)
        goals = sample_df[self.goal_col].to_numpy(dtype=np.float32)
        rewards_s = sample_df[self.reward_col].to_numpy(dtype=np.float32)
        best_idxs = sample_df[self.best_idx_col].to_numpy(dtype=np.float32) if self.best_idx_col in sample_df else None
        epochs = sample_df[self.epoch_col].to_numpy(dtype=np.float32) if self.epoch_col in sample_df else None

        return {
            "states": states,
            "actions": actions,
            "goals": goals,
            "rewards": rewards_s,
            "indices": idxs,
            "best_idx": best_idxs,
            "epochs": epochs,
        }

    def sample_target_goal(self, n, target_goal, strategy='meaningful_bias'):
        """
        Sample n transitions from buffer where goal matches target_goal.
        Used after exploration phase to focus training on target goal only.

        Args:
            n: number of samples
            target_goal: int, goal index to filter by (e.g., args.ig)
            strategy: 'uniform' or 'meaningful_bias'

        Returns:
            dict with sampled batch, filtered to target goal only
        """
        if not self.data:
            raise ValueError("Buffer is empty; cannot sample.")

        df = pd.DataFrame(self.data)

        # Filter to only transitions with the target goal
        target_mask = (df[self.goal_col] == target_goal)
        target_df = df[target_mask]

        if len(target_df) == 0:
            raise ValueError(f"No transitions found for target goal {target_goal} in buffer")

        # Get indices in the filtered dataframe
        target_indices = target_df.index.to_numpy()

        if len(target_indices) < n:
            print(f" WARNING: Only {len(target_indices)} target goal transitions available, requested {n}. Sampling with replacement.")
            idxs = np.random.choice(target_indices, size=n, replace=True)

        else:
            if strategy == 'meaningful_bias':
                # Extract rewards for target goal transitions
                rewards = target_df[self.reward_col].to_numpy(dtype=np.float32)
                success_mask = (rewards == 0.0)

                if success_mask.any():
                    # Get actual buffer indices of successes
                    success_indices = target_indices[success_mask]

                    # Sample 1 guaranteed success
                    forced_success_idx = np.random.choice(success_indices, size=1)

                    # Sample (n-1) random from all target goal transitions
                    remaining_idxs = np.random.choice(target_indices, size=n-1, replace=False)

                    # Combine: [1 success, n-1 random target goal samples]
                    idxs = np.concatenate([forced_success_idx, remaining_idxs])

                    print(f" TARGET Goal Buffer: 1 success + {n-1} random samples (from {len(success_indices)} target successes available)")
                else:
                    # No successes for target goal yet - sample randomly from target goal
                    idxs = np.random.choice(target_indices, size=n, replace=False)
                    print(f" WARNING: No successes yet for target goal {target_goal} - random sampling ({n} samples)")

            else:  # uniform
                # Uniform sampling from target goal transitions only
                idxs = np.random.choice(target_indices, size=n, replace=False)

        # Extract sampled data using the same pattern as sample()
        sample_df = df.iloc[idxs]

        states = sample_df[self.state_cols].to_numpy(dtype=np.float32)
        actions = sample_df[self.action_cols].to_numpy(dtype=np.float32)
        goals = sample_df[self.goal_col].to_numpy(dtype=np.float32)
        rewards_s = sample_df[self.reward_col].to_numpy(dtype=np.float32)
        best_idxs = sample_df[self.best_idx_col].to_numpy(dtype=np.float32) if self.best_idx_col in sample_df else None
        epochs = sample_df[self.epoch_col].to_numpy(dtype=np.float32) if self.epoch_col in sample_df else None

        return {
            "states": states,
            "actions": actions,
            "goals": goals,
            "rewards": rewards_s,
            "indices": idxs,
            "best_idx": best_idxs,
            "epochs": epochs,
        }

    def plot_success_rate(self, target_goal, batch_size, output_dir):
        """
        Plot success rate per epoch for a specific target goal.
        Reads data directly from buffer.csv if not already loaded.

        Args:
            target_goal (int): Goal index to track (e.g., 0 for SC, 1 for OHC)
            batch_size (int): Number of actions per epoch (for computing success rate %)
            output_dir (str): Directory to save the plot
        """
        # Load buffer from CSV if not already in memory
        if not self.data:
            if os.path.exists(self.buffer_path):
                print(f"  Loading buffer from {self.buffer_path} for plotting...")
                self.load()
            else:
                print(f"Buffer file not found at {self.buffer_path}, cannot plot success rate")
                return

        df = pd.DataFrame(self.data)

        # Group by epoch and count successes for target goal
        # Success = reward == 0 AND goal == target_goal
        success_counts = []
        epochs_list = []

        for epoch in sorted(df[self.epoch_col].unique()):
            epoch_data = df[df[self.epoch_col] == epoch]
            # Count transitions with reward=0 AND goal=target_goal
            successes = ((epoch_data[self.reward_col] == 0.0) &
                        (epoch_data[self.goal_col] == target_goal)).sum()
            success_rate = (successes / batch_size) * 100.0  # Convert to percentage

            success_counts.append(success_rate)
            epochs_list.append(epoch)

        # Create plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_list, success_counts, marker='o', linestyle='--',
                color='blue', linewidth=2, markersize=6)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Success Rate (%)", fontsize=12)
        plt.title(f"Success Rate for Goal {target_goal} (Target Structure)", fontsize=14)
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"success_rate_goal_{target_goal}.svg")
        plt.savefig(save_path)
        plt.close()
        print(f"  OK: Saved success rate plot to: {save_path}")

    def plot_action_uncertainty(self, output_dir):
        """
        Plot mean ± std of actions over epochs (2 subplots: sigmas and lambdas).
        Shows how actions become more deterministic as training progresses.
        Reads data directly from buffer.csv if not already loaded.
        Args:
            output_dir (str): Directory to save the plot
        """
        # Load buffer from CSV if not already in memory
        if not self.data:
            if os.path.exists(self.buffer_path):
                print(f"  Loading buffer from {self.buffer_path} for plotting...")
                self.load()
            else:
                print(f"Buffer file not found at {self.buffer_path}, cannot plot action uncertainty")
                return
        
        df = pd.DataFrame(self.data)
        
        # Compute mean and std per epoch for each action dimension
        # action_0 = σ_II, action_1 = σ_AB (fixed at 1), action_2 = λ_II, action_3 = λ_IJ
        action_dims = [0, 1, 2, 3]
        action_names = ['σ_II', 'σ_AB', 'λ_II', 'λ_IJ']
        
        epochs_list = sorted(df[self.epoch_col].unique())
        means = {dim: [] for dim in action_dims}
        stds = {dim: [] for dim in action_dims}
        
        for epoch in epochs_list:
            epoch_data = df[df[self.epoch_col] == epoch]
            for dim in action_dims:
                action_col = self.action_cols[dim]
                means[dim].append(epoch_data[action_col].mean())
                stds[dim].append(epoch_data[action_col].std())
        
        # Create 2 subplots: one for sigmas, one for lambdas
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # Subplot 1: Sigmas (σ_II, σ_AB)
        ax = axes[0]
        sigma_dims = [0, 1]
        sigma_names = ['σ_II', 'σ_AB']
        colors_sigma = ['#d95f02', '#e7298a']  # Orange, pink
        markers_sigma = ['o', 'x']
        
        for dim, name, color, marker in zip(sigma_dims, sigma_names, colors_sigma, markers_sigma):
            mean_vals = np.array(means[dim])
            std_vals = np.array(stds[dim])
            
            ax.plot(epochs_list, mean_vals, marker=marker, color=color, 
                    linewidth=2, markersize=6, label=name)
            # ax.fill_between(epochs_list, mean_vals - std_vals, mean_vals + std_vals, 
            #                 color=color, alpha=0.15)
        
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Average Action", fontsize=11)
        ax.set_title("Sigmas per Epoch", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Subplot 2: Lambdas (λ_II, λ_IJ)
        ax = axes[1]
        lambda_dims = [2, 3]
        lambda_names = ['λ_II', 'λ_IJ']
        colors_lambda = ['#1b9e77', '#7570b3']  # Teal, purple
        markers_lambda = ['s', '^']
        
        for dim, name, color, marker in zip(lambda_dims, lambda_names, colors_lambda, markers_lambda):
            mean_vals = np.array(means[dim])
            std_vals = np.array(stds[dim])
            
            ax.plot(epochs_list, mean_vals, marker=marker, color=color, 
                    linewidth=2, markersize=6, label=name)
            # ax.fill_between(epochs_list, mean_vals - std_vals, mean_vals + std_vals, 
            #                 color=color, alpha=0.15)



        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Average Action", fontsize=11)
        ax.set_title("Lambdas per Epoch", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Subplot 3: Std as % of action range (per epoch)
        ax = axes[2]
        std_dims = [0, 2, 3]  # Skip σ_AB which is fixed at 1.0

        # Normalize std to % of allowed action range for a cleaner, unitless view
        action_ranges = np.asarray(self.u_max) - np.asarray(self.u_min)
        action_ranges = np.where(action_ranges == 0, 1.0, action_ranges)  # avoid divide-by-zero

        std_pct_per_dim = []
        for dim in std_dims:
            std_vals = np.array(stds[dim])
            std_pct = (std_vals / action_ranges[dim]) * 100.0
            std_pct_per_dim.append(std_pct)

        # Average std across all action dimensions to summarize overall uncertainty
        system_std_pct = np.mean(np.stack(std_pct_per_dim, axis=0), axis=0)

        ax.plot(epochs_list, system_std_pct, marker='o', color='#1f78b4',
                linewidth=2, markersize=6, label='Std of system')

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Avg Std (% of range)", fontsize=11)
        ax.set_title("System Std per Epoch (percentage)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "avg_actions_per_epoch.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"  OK: Saved action uncertainty plot to: {save_path}")

    def plot_q_value_progression(self, output_dir):
        """
        Plot mean reward per epoch (proxy for Q-value, shows critic learning).
        Reads data directly from buffer.csv if not already loaded.

        Computes mean reward per epoch from buffer as a proxy for Q-function performance.
        Since rewards are 0 (success) or -1 (failure), mean reward indicates success rate.

        Args:
            output_dir (str): Directory to save the plot
        """
        # Load buffer from CSV if not already in memory
        if not self.data:
            if os.path.exists(self.buffer_path):
                print(f"  Loading buffer from {self.buffer_path} for plotting...")
                self.load()
            else:
                print(f"Buffer file not found at {self.buffer_path}, cannot plot Q-value progression")
                return

        df = pd.DataFrame(self.data)

        # Compute mean reward per epoch as proxy for Q-value
        epochs_list = sorted(df[self.epoch_col].unique())
        mean_rewards = []

        for epoch in epochs_list:
            epoch_data = df[df[self.epoch_col] == epoch]
            mean_reward = epoch_data[self.reward_col].mean()
            mean_rewards.append(mean_reward)

        # Create plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_list, mean_rewards, marker='o', linestyle='--',
                color='purple', linewidth=2, markersize=6)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Mean Reward (Proxy for Q-Value)", fontsize=12)
        plt.title("Q-Function Learning Progression (Mean Reward per Epoch)", fontsize=14)
        plt.ylim(-1.05, 0.05)
        plt.axhline(y=0.0, color='gray', linestyle=':', linewidth=1, label='Target (All Successes)')
        plt.axhline(y=-1.0, color='gray', linestyle=':', linewidth=1, label='Baseline (All Failures)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "q_value_progression.svg")
        plt.savefig(save_path)
        plt.close()
        print(f"  OK: Saved Q-value progression plot to: {save_path}")

