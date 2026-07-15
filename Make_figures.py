"""
Run Visualization
==================
Visualizes training results from a single Run_Policy.py run.
Produces a 2×2 figure per structure:
  top-left:     λ_ii vs epoch
  top-right:    σ_ii vs epoch
  bottom-left:  λ_ij vs epoch
  bottom-right: mean reward (left y-axis) + crystal quality (right y-axis)

Usage:
    python Make_figures.py --run_dir ./Model_and_Results --str_index 4 --three_d 0
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (mirror Run_Policy.py)
# ══════════════════════════════════════════════════════════════════════════════

STRUCTURE_DICT = {
    0: "FCC", 1: "HCP", 2: "BCC", 3: "ICO", 4: "SC",
    5: "Cub_Diam", 6: "Hex_Diam", 7: "OHC", 8: "SSS", 9: "BTr", 10: "Other"
}

GOAL_DIC_2D = {"SC": 0.64, "OHC": 0.72, "SSS": 0.68, "BTr": 0.48}
GOAL_DIC_3D = {
    "FCC": 0.47, "HCP": 0.9, "BCC": 0.559, "ICO": 0.8,
    "SC": 0.646, "Cub_Diam": 0.486, "Hex_Diam": 0.519
}

LABEL_MAP = {
    "FCC": "Face-Centered Cubic", "HCP": "Hexagonal Close-Packed",
    "BCC": "Body-Centered Cubic", "ICO": "Icosahedral",
    "SC": "Simple Cubic", "Cub_Diam": "Cubic Diamond",
    "Hex_Diam": "Hexagonal Diamond", "OHC": "Open Honeycomb",
    "SSS": "Square Single Stripe", "BTr": "Binary Kagome Triangle",
    "Other": "Other"
}

# Base colors matching create_appendix_figures_unified.py STRUCTURES_CONFIG
# Keyed by structure name; unknown structures fall back to a neutral grey.
BASE_COLORS = {
    "SC":       "#d62728",
    "BCC":      "#2ca02c",
    "SSS":      "#9467bd",
    "BTr":      "#ff7f0e",
    "OHC":      "#1f77b4",
    "Cub_Diam": "#17becf",
    "Hex_Diam": "#17becf",
    "FCC":      "#8c564b",
    "HCP":      "#e377c2",
    "ICO":      "#7f7f7f",
    "Other":    "#aec7e8",
}

ACTION_LABELS = {
    "action_0": r"$\sigma_{ii}$",
    "action_2": r"$\lambda_{ii}$",
    "action_3": r"$\lambda_{ij}$",
}

AXIS_LIMITS = {
    "action_0": (0.3, 2.5),
    "action_2": (0.0, 3.0),
    "action_3": (0.0, 3.0),
}

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_run_config(str_index, three_d):
    """Derive per-run visualization config from Run_Policy hyperparameters."""
    structure_name = STRUCTURE_DICT[str_index]
    goal_dic = GOAL_DIC_2D if three_d == 0 else GOAL_DIC_3D

    if structure_name not in goal_dic:
        raise ValueError(
            f"Structure '{structure_name}' (str_index={str_index}) not in goal_dic "
            f"for three_d={three_d}. Available: {list(goal_dic.keys())}"
        )

    key_list = list(goal_dic.keys())
    ig = key_list.index(structure_name)

    return {
        "structure_name": structure_name,
        "label": LABEL_MAP.get(structure_name, structure_name),
        "goal_index": ig,
        "goal": goal_dic[structure_name],
        "color": BASE_COLORS.get(structure_name, "#aec7e8"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# COLOR UTILITIES  (ported from create_appendix_figures_unified.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_dark_light_colors(base_hex_color, structure_name=None):
    """
    Return (dark_color, light_color) for dual y-axis plotting.
    Structure-specific assignments match create_appendix_figures_unified.py exactly.
    """
    if structure_name == "SL":
        base_rgb = mcolors.to_rgb(base_hex_color)
        base_hsv = mcolors.rgb_to_hsv(base_rgb)
        dark_hsv = (base_hsv[0], min(1.0, base_hsv[1] * 1.3), base_hsv[2] * 0.65)
        return mcolors.hsv_to_rgb(dark_hsv), "#D20000"

    if structure_name == "OHC":
        return "#000080", "#1E90FF"

    if structure_name == "BTr":
        return "#FF7518", "#FFA500"

    if structure_name == "SC":
        return "#D20000", "#FF2020"

    if structure_name == "BCC":
        return "#006241", "#3CB371"

    if structure_name == "SSS":
        return "#8B008B", "#FF1DCE"

    if structure_name in ("Diamond", "Cub_Diam", "Hex_Diam"):
        base_rgb = mcolors.to_rgb(base_hex_color)
        base_hsv = mcolors.rgb_to_hsv(base_rgb)
        dark_hsv = (base_hsv[0], min(1.0, base_hsv[1] * 1.3), base_hsv[2] * 0.65)
        return mcolors.hsv_to_rgb(dark_hsv), "#0CAFFF"

    # General case
    base_rgb = mcolors.to_rgb(base_hex_color)
    base_hsv = mcolors.rgb_to_hsv(base_rgb)
    dark_hsv  = (base_hsv[0], min(1.0, base_hsv[1] * 1.3), base_hsv[2] * 0.65)
    light_hsv = (base_hsv[0], base_hsv[1] * 0.5, min(1.0, base_hsv[2] * 1.25 + 0.15))
    return mcolors.hsv_to_rgb(dark_hsv), mcolors.hsv_to_rgb(light_hsv)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING STYLE
# ══════════════════════════════════════════════════════════════════════════════

def setup_style():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_buffer_data(run_dir, config):
    """Load and aggregate buffer data from a Run_Policy run directory."""
    run_dir = Path(run_dir)
    buffer_file = run_dir / "1_data" / "training_dynamics" / "buffer.csv"
    goal_index = config["goal_index"]

    if not buffer_file.exists():
        print(f"  Warning: Buffer file not found: {buffer_file}")
        return None, None, None

    df = pd.read_csv(buffer_file)
    df_goal = df[(df["goal"] == goal_index) & (df["epoch"] >= 1)].copy()

    if len(df_goal) == 0:
        print(f"  No transitions for goal index {goal_index} in buffer")
        return None, None, None

    print(f"  Loaded {len(df_goal)} transitions for {config['structure_name']} (goal={goal_index})")

    # Mean actions per epoch
    actions_per_epoch = []
    for epoch in sorted(df_goal["epoch"].unique()):
        epoch_data = df_goal[df_goal["epoch"] == epoch]
        actions_per_epoch.append({
            "epoch": epoch,
            "action_0": epoch_data["action_0"].mean(),
            "action_2": epoch_data["action_2"].mean(),
            "action_3": epoch_data["action_3"].mean(),
        })
    actions_df = pd.DataFrame(actions_per_epoch)

    # Mean reward per epoch
    rewards_df = df_goal.groupby("epoch")["reward"].mean().reset_index()
    rewards_df.columns = ["epoch", "mean_reward"]

    # Mean crystal quality (normalized by goal fraction) per epoch
    fractions_df = None
    next_state_col = f"next_state_{goal_index}"
    if next_state_col in df_goal.columns:
        fractions_df = df_goal.groupby("epoch")[next_state_col].mean().reset_index()
        fractions_df.columns = ["epoch", "mean_fraction"]
        fractions_df["mean_fraction"] /= config["goal"]

    return actions_df, rewards_df, fractions_df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def create_figure(run_dir, config, output_dir):
    """
    Create a 2×2 figure for the target structure:
      top-left:     λ_ii vs epoch
      top-right:    σ_ii vs epoch
      bottom-left:  λ_ij vs epoch
      bottom-right: mean reward (left y-axis, dark) + crystal quality (right y-axis, light)
    """
    setup_style()
    run_dir        = Path(run_dir)
    output_dir     = Path(output_dir)
    structure_name = config["structure_name"]
    param_color    = config["color"]
    dark_color, light_color = get_dark_light_colors(param_color, structure_name)

    print(f"\nProcessing {config['label']} (goal_index={config['goal_index']})...")

    actions_df, rewards_df, fractions_df = load_buffer_data(run_dir, config)
    if actions_df is None or rewards_df is None:
        print(f"  Warning: Failed to load data — aborting figure")
        return None

    epochs     = actions_df["epoch"].values
    sigma_aa   = actions_df["action_0"].values
    lambda_aa  = actions_df["action_2"].values
    lambda_ab  = actions_df["action_3"].values

    # Prepend epoch 0 with initial state (midpoint of action space)
    init_sigma     = (AXIS_LIMITS["action_0"][0] + AXIS_LIMITS["action_0"][1]) / 2
    init_lambda_aa = (AXIS_LIMITS["action_2"][0] + AXIS_LIMITS["action_2"][1]) / 2
    init_lambda_ab = (AXIS_LIMITS["action_3"][0] + AXIS_LIMITS["action_3"][1]) / 2

    epochs_with_init     = np.insert(epochs, 0, 0)
    sigma_with_init      = np.insert(sigma_aa,  0, init_sigma)
    lambda_aa_with_init  = np.insert(lambda_aa, 0, init_lambda_aa)
    lambda_ab_with_init  = np.insert(lambda_ab, 0, init_lambda_ab)

    # Prepend synthetic epoch-0 point: reward=-1 (no success), crystal quality=0
    rewards_df = pd.concat([
        pd.DataFrame({"epoch": [0], "mean_reward": [-1.0]}),
        rewards_df
    ], ignore_index=True)
    if fractions_df is not None:
        fractions_df = pd.concat([
            pd.DataFrame({"epoch": [0], "mean_fraction": [0.0]}),
            fractions_df
        ], ignore_index=True)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_width  = 8.27   # A4 width
    fig_height = 5.0
    fig = plt.figure(figsize=(fig_width, fig_height))

    plot_width   = 0.38
    plot_height  = 0.36
    h_spacing    = 0.08
    v_spacing    = 0.08
    left_margin  = 0.10
    bottom_margin = 0.15

    x_lim = (-0.5, epochs[-1] + 0.5)

    # ── Shared spine style helper ─────────────────────────────────────────────
    def style_ax(ax, hide_xticks=False):
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        if hide_xticks:
            ax.tick_params(axis='x', labelbottom=False)

    # ── Top-left: λ_ii ────────────────────────────────────────────────────────
    ax_lambda_aa = fig.add_axes([
        left_margin,
        bottom_margin + plot_height + v_spacing,
        plot_width, plot_height
    ])
    ax_lambda_aa.plot(epochs_with_init, lambda_aa_with_init,
                      color=param_color, linewidth=1.2, alpha=0.8)
    ax_lambda_aa.scatter(epochs_with_init, lambda_aa_with_init,
                         color=param_color, s=25, alpha=0.8,
                         edgecolors='black', linewidths=0.4, marker='s')
    ax_lambda_aa.set_ylabel(r"$\lambda_{ii}$", fontsize=14, color='black')
    ax_lambda_aa.tick_params(axis='y', labelsize=10, labelcolor='black', colors='black', pad=2)
    style_ax(ax_lambda_aa, hide_xticks=True)
    ax_lambda_aa.set_xlim(*x_lim)
    ax_lambda_aa.set_ylim(0.0, 3.15)
    ax_lambda_aa.set_yticks([0.0, 1.0, 2.0, 3.0])

    # ── Top-right: σ_ii ───────────────────────────────────────────────────────
    ax_sigma = fig.add_axes([
        left_margin + plot_width + h_spacing,
        bottom_margin + plot_height + v_spacing,
        plot_width, plot_height
    ])
    ax_sigma.plot(epochs_with_init, sigma_with_init,
                  color=param_color, linewidth=1.2, alpha=0.8)
    ax_sigma.scatter(epochs_with_init, sigma_with_init,
                     color=param_color, s=25, alpha=0.8,
                     edgecolors='black', linewidths=0.4, marker='o')
    ax_sigma.set_ylabel(r"$\sigma_{ii}$", fontsize=14, color='black')
    ax_sigma.tick_params(axis='y', labelsize=10, labelcolor='black', colors='black', pad=2)
    style_ax(ax_sigma, hide_xticks=True)
    ax_sigma.set_xlim(*x_lim)
    ax_sigma.set_ylim(0.3, 2.5)
    ax_sigma.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])

    # ── Bottom-left: λ_ij ─────────────────────────────────────────────────────
    ax_lambda_ab = fig.add_axes([
        left_margin,
        bottom_margin,
        plot_width, plot_height
    ])
    ax_lambda_ab.plot(epochs_with_init, lambda_ab_with_init,
                      color=param_color, linewidth=1.2, alpha=0.8)
    ax_lambda_ab.scatter(epochs_with_init, lambda_ab_with_init,
                         color=param_color, s=25, alpha=0.8,
                         edgecolors='black', linewidths=0.4, marker='^')
    ax_lambda_ab.set_ylabel(r"$\lambda_{ij}$", fontsize=14, color='black')
    ax_lambda_ab.set_xlabel("Epoch", fontsize=11)
    ax_lambda_ab.tick_params(axis='y', labelsize=10, labelcolor='black', colors='black', pad=2)
    ax_lambda_ab.tick_params(axis='x', labelsize=10)
    style_ax(ax_lambda_ab)
    ax_lambda_ab.set_xlim(*x_lim)
    ax_lambda_ab.set_ylim(0.0, 3.15)
    ax_lambda_ab.set_yticks([0.0, 1.0, 2.0, 3.0])

    # ── Bottom-right: unified rewards + crystal quality ────────────────────────
    ax_unified = fig.add_axes([
        left_margin + plot_width + h_spacing,
        bottom_margin,
        plot_width, plot_height
    ])

    epochs_reward = rewards_df["epoch"].values
    rewards       = rewards_df["mean_reward"].values

    ax_unified.set_xlabel("Epoch", fontsize=11)
    ax_unified.set_ylabel("Mean Reward", fontsize=10, color=dark_color)
    ax_unified.scatter(epochs_reward, rewards,
                       color=dark_color, s=25, alpha=0.8,
                       edgecolors='black', linewidths=0.3, marker='o',
                       label='Mean Reward')
    ax_unified.tick_params(axis='y', labelsize=10, labelcolor=dark_color, colors='black')
    ax_unified.tick_params(axis='x', labelsize=10)
    ax_unified.spines['left'].set_color('black')
    ax_unified.spines['left'].set_linewidth(1.5)
    ax_unified.spines['top'].set_visible(False)
    ax_unified.set_xlim(*x_lim)

    if fractions_df is not None:
        ax2 = ax_unified.twinx()
        epochs_frac   = fractions_df["epoch"].values
        mean_fraction = fractions_df["mean_fraction"].values

        ax2.set_ylabel("Average Crystal Quality", fontsize=10,
                       color=light_color, rotation=-90, labelpad=15)
        ax2.scatter(epochs_frac, mean_fraction,
                    color=light_color, s=25, alpha=0.8,
                    edgecolors='black', linewidths=0.3, marker='s',
                    label='Crystal Quality')
        ax2.tick_params(axis='y', labelsize=10, labelcolor=light_color, colors='black')
        ax2.spines['right'].set_color('black')
        ax2.spines['right'].set_linewidth(1.5)
        ax2.set_ylim(-0.05, 1.05)

        lines1, labels1 = ax_unified.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax_unified.legend(lines1 + lines2, labels1 + labels2,
                              loc='best', fontsize=7, frameon=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ("pdf", "svg"):
        out = output_dir / f"{structure_name}_unified.{fmt}"
        fig.savefig(out, format=fmt, dpi=300, bbox_inches='tight')
        print(f"  Saved {fmt.upper()}: {out}")

    plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualize a Run_Policy.py training run")
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to the Model_and_Results directory of the run')
    parser.add_argument('--str_index', type=int, default=4,
                        help='Target structure index (same as Run_Policy --str_index)')
    parser.add_argument('--three_d', type=int, default=0,
                        help='0=2D goal_dic, 1=3D goal_dic (same as Run_Policy --three_d)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: <run_dir>/4_plots)')
    args = parser.parse_args()

    config = build_run_config(args.str_index, args.three_d)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.run_dir) / "4_plots"
    )

    print("=" * 70)
    print("RUN VISUALIZATION")
    print("=" * 70)
    print(f"Run directory:  {args.run_dir}")
    print(f"Structure:      {config['label']} ({config['structure_name']})")
    print(f"Goal index:     {config['goal_index']}")
    print(f"Goal threshold: {config['goal']}")
    print(f"Color:          {config['color']}")
    print(f"Output:         {output_dir}")
    print("=" * 70)

    create_figure(args.run_dir, config, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
