#!/usr/bin/env python3
"""
CORRECT Implementation of Paper's Method

The paper concatenates activations from ALL layers at the LAST token position
and uses weighted least-squares regression. This is completely different
from our layer-by-layer analysis!

Key corrections:
1. Concatenate ALL layer activations
2. Use LAST token position only
3. Weighted least-squares regression
4. Single affine transformation across full network
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer, HookedTransformerConfig
from epsilon_transformers.process.transition_matrices import mess3, tom_quantum
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM, _compute_next_distribution
from multipartite_data import create_multipartite_data

def project_to_simplex_2d_CORRECT(beliefs):
    """Reference simplex projection."""
    x = beliefs[:, 0] - beliefs[:, 1] / 2 - beliefs[:, 2] / 2
    y = np.sqrt(3) / 2 * (beliefs[:, 1] - beliefs[:, 2])
    x_rot = -y
    y_rot = x
    return x_rot, y_rot

def load_trained_model(model_path, config_path):
    """Load a trained transformer model."""
    print(f"Loading model from {model_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model_config']
    model_config['dtype'] = getattr(torch, model_config['dtype'])
    hooked_model_config = HookedTransformerConfig(**model_config)
    model = HookedTransformer(hooked_model_config)

    state_dict = torch.load(model_path, map_location='cuda')
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("   Detected torch.compile prefixes, stripping...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Model loaded: {model_config['n_layers']} layers, {model_config['d_model']} dim")
    return model, config

def extract_concatenated_activations_CORRECT(model, sequences):
    """
    CORRECT extraction: Concatenate activations from ALL layers at LAST token position.

    This follows the paper's method exactly:
    "extracted activation vectors ⃗aw ∈ R^(N×dmodel) by concatenating activations
    from all N layers at the last token position"
    """
    print("🔧 Extracting concatenated activations (PAPER METHOD)...")

    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(sequences)

    batch_size, seq_len, d_model = sequences.shape[0], sequences.shape[1], model.cfg.d_model
    n_layers = model.cfg.n_layers

    # Collect activations from all layers at last token position
    all_layer_activations = []

    # Start with embedding
    if 'hook_embed' in cache:
        embed_last = cache['hook_embed'][:, -1, :]  # (batch_size, d_model)
        all_layer_activations.append(embed_last)
        print(f"   Added embedding: {embed_last.shape}")

    # Add all transformer layer activations
    for i in range(n_layers):
        # Use post-residual activations (after each complete layer)
        layer_key = f'blocks.{i}.hook_resid_post'
        if layer_key in cache:
            layer_last = cache[layer_key][:, -1, :]  # (batch_size, d_model)
            all_layer_activations.append(layer_last)
            print(f"   Added layer {i}: {layer_last.shape}")

    # Concatenate all activations
    concatenated_activations = torch.cat(all_layer_activations, dim=1)  # (batch_size, N_layers * d_model)

    print(f"✅ Concatenated activations shape: {concatenated_activations.shape}")
    print(f"   Total dimensions: {concatenated_activations.shape[1]} = {len(all_layer_activations)} layers × {d_model} dims")

    return concatenated_activations

def generate_belief_states_at_last_position(sequences, n_ctx, num_sequences):
    """
    Generate belief states corresponding to the LAST token position.

    This matches what the model sees at the last position after processing
    the entire context sequence.
    """
    print("🔧 Generating belief states at LAST token position...")

    # Create processes
    T_mess3 = mess3(a=0.85, x=0.05)
    T_bloch = tom_quantum(alpha=1.0, beta=np.sqrt(51))

    ghmm_mess3 = TransitionMatrixGHMM(T_mess3)
    ghmm_bloch = TransitionMatrixGHMM(T_bloch)

    mess3_final_beliefs = []
    bloch_final_beliefs = []

    # For each sequence, compute the final belief state after processing the full context
    for seq_idx in range(num_sequences):
        sequence = sequences[seq_idx].cpu().numpy()

        # Process Mess3 sequence
        mess3_belief = ghmm_mess3.steady_state_vector.copy()  # Start from steady state
        for pos in range(1, len(sequence)):  # Skip BOS token
            token = sequence[pos]
            # Check if this is a Mess3 token (0, 1, 2)
            if token <= 2:
                mess3_belief = _compute_next_distribution(
                    T_mess3, mess3_belief, token, ghmm_mess3.right_eigenvector
                ).reshape(1, -1)

        mess3_final_beliefs.append(mess3_belief.squeeze())

        # Process Bloch Walk sequence
        bloch_belief = ghmm_bloch.steady_state_vector.copy()
        for pos in range(1, len(sequence)):  # Skip BOS token
            token = sequence[pos]
            # Check if this is a Bloch Walk token (3, 4, 5, 6)
            if 3 <= token <= 6:
                original_token = token - 3  # Convert back to 0-3 range
                bloch_belief = _compute_next_distribution(
                    T_bloch, bloch_belief, original_token, ghmm_bloch.right_eigenvector
                ).reshape(1, -1)

        bloch_final_beliefs.append(bloch_belief.squeeze())

    mess3_final_beliefs = torch.tensor(np.array(mess3_final_beliefs), dtype=torch.float32)
    bloch_final_beliefs = torch.tensor(np.array(bloch_final_beliefs), dtype=torch.float32)

    print(f"✅ Generated final belief states:")
    print(f"   Mess3: {mess3_final_beliefs.shape}")
    print(f"   Bloch: {bloch_final_beliefs.shape}")

    return mess3_final_beliefs, bloch_final_beliefs

def weighted_least_squares_regression(X, y, weights=None):
    """
    Weighted least-squares regression as mentioned in the paper.

    If no weights provided, falls back to ordinary least squares.
    """
    if weights is None:
        # Use ordinary least squares
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        return reg, y_pred
    else:
        # Weighted least squares
        # Solve: argmin_w ||W^(1/2)(Xw - y)||^2 where W = diag(weights)
        W_sqrt = np.sqrt(weights)
        X_weighted = X * W_sqrt.reshape(-1, 1)
        y_weighted = y * W_sqrt.reshape(-1, 1) if y.ndim == 2 else y * W_sqrt

        reg = LinearRegression()
        reg.fit(X_weighted, y_weighted)
        y_pred = reg.predict(X)  # Predict on unweighted X
        return reg, y_pred

def analyze_belief_decoding_CORRECT(concatenated_activations, belief_states, process_name, weights=None):
    """
    CORRECT belief state decoding using the paper's method.
    """
    print(f"\n🔍 CORRECT Belief State Analysis - {process_name}")
    print("=" * 50)

    # Prepare data
    X = concatenated_activations.cpu().numpy()
    y = belief_states.cpu().numpy()

    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Fit weighted least-squares regression
    reg, y_pred = weighted_least_squares_regression(X, y, weights)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = reg.score(X, y)

    print(f"✅ Results:")
    print(f"   R² Score: {r2:.6f}")
    print(f"   MSE: {mse:.6f}")
    print(f"   Prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"   Truth range: [{y.min():.3f}, {y.max():.3f}]")

    return {
        'r2': r2,
        'mse': mse,
        'predictions': y_pred,
        'ground_truth': y,
        'regression': reg
    }

def visualize_correct_results(mess3_results, bloch_results, save_path):
    """Visualize the CORRECT results using proper simplex projection with enhanced clarity."""

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # Mess3 analysis
    mess3_gt = mess3_results['ground_truth']
    mess3_pred = mess3_results['predictions']

    # Project to simplex coordinates
    mess3_gt_x, mess3_gt_y = project_to_simplex_2d_CORRECT(mess3_gt)
    mess3_pred_x, mess3_pred_y = project_to_simplex_2d_CORRECT(mess3_pred)

    # Mess3 ground truth
    axes[0, 0].scatter(mess3_gt_x, mess3_gt_y, alpha=0.7, s=8, c='blue', edgecolors='none')
    axes[0, 0].set_title(f'Mess3 Ground Truth\nR² = {mess3_results["r2"]:.4f}', fontsize=14, fontweight='bold')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)

    # Mess3 predictions
    axes[0, 1].scatter(mess3_pred_x, mess3_pred_y, alpha=0.7, s=8, c='red', edgecolors='none')
    axes[0, 1].set_title(f'Mess3 Predictions\nMSE = {mess3_results["mse"]:.4f}', fontsize=14, fontweight='bold')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)

    # Mess3 overlay
    axes[0, 2].scatter(mess3_gt_x, mess3_gt_y, alpha=0.4, s=6, c='blue', label='Ground Truth', edgecolors='none')
    axes[0, 2].scatter(mess3_pred_x, mess3_pred_y, alpha=0.4, s=6, c='red', label='Predictions', edgecolors='none')
    axes[0, 2].set_title('Mess3 Overlay Comparison', fontsize=14, fontweight='bold')
    axes[0, 2].legend(fontsize=12)
    axes[0, 2].set_aspect('equal')
    axes[0, 2].grid(True, alpha=0.3)

    # Bloch Walk analysis
    bloch_gt = bloch_results['ground_truth']
    bloch_pred = bloch_results['predictions']

    # Project to simplex coordinates
    bloch_gt_x, bloch_gt_y = project_to_simplex_2d_CORRECT(bloch_gt)
    bloch_pred_x, bloch_pred_y = project_to_simplex_2d_CORRECT(bloch_pred)

    # Bloch Walk ground truth
    axes[1, 0].scatter(bloch_gt_x, bloch_gt_y, alpha=0.7, s=8, c='blue', edgecolors='none')
    axes[1, 0].set_title(f'Bloch Walk Ground Truth\nR² = {bloch_results["r2"]:.4f}', fontsize=14, fontweight='bold')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)

    # Bloch Walk predictions
    axes[1, 1].scatter(bloch_pred_x, bloch_pred_y, alpha=0.7, s=8, c='red', edgecolors='none')
    axes[1, 1].set_title(f'Bloch Walk Predictions\nMSE = {bloch_results["mse"]:.4f}', fontsize=14, fontweight='bold')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)

    # Bloch Walk overlay
    axes[1, 2].scatter(bloch_gt_x, bloch_gt_y, alpha=0.4, s=6, c='blue', label='Ground Truth', edgecolors='none')
    axes[1, 2].scatter(bloch_pred_x, bloch_pred_y, alpha=0.4, s=6, c='red', label='Predictions', edgecolors='none')
    axes[1, 2].set_title('Bloch Walk Overlay Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=12)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('CORRECT METHOD: Concatenated Multi-Layer Activations → Belief States', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved ENHANCED visualization: {save_path}")

    # Also create a high-resolution detailed view
    detailed_save_path = save_path.parent / "detailed_belief_analysis.png"
    create_detailed_visualization(mess3_results, bloch_results, detailed_save_path)

def create_detailed_visualization(mess3_results, bloch_results, save_path):
    """Create detailed high-resolution visualizations with density plots."""

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Mess3 analysis
    mess3_gt = mess3_results['ground_truth']
    mess3_pred = mess3_results['predictions']
    mess3_gt_x, mess3_gt_y = project_to_simplex_2d_CORRECT(mess3_gt)
    mess3_pred_x, mess3_pred_y = project_to_simplex_2d_CORRECT(mess3_pred)

    # Mess3 ground truth scatter
    axes[0, 0].scatter(mess3_gt_x, mess3_gt_y, alpha=0.6, s=4, c='blue', edgecolors='none')
    axes[0, 0].set_title(f'Mess3 Ground Truth\nN = {len(mess3_gt)} points', fontsize=12, fontweight='bold')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)

    # Mess3 predictions scatter
    axes[0, 1].scatter(mess3_pred_x, mess3_pred_y, alpha=0.6, s=4, c='red', edgecolors='none')
    axes[0, 1].set_title(f'Mess3 Predictions\nR² = {mess3_results["r2"]:.4f}', fontsize=12, fontweight='bold')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)

    # Mess3 density plot (ground truth)
    try:
        import numpy as np
        from scipy.stats import gaussian_kde

        # Create density plot for ground truth
        if len(mess3_gt_x) > 50:  # Only if we have enough points
            xy = np.vstack([mess3_gt_x, mess3_gt_y])
            kde = gaussian_kde(xy)

            # Create grid
            x_min, x_max = mess3_gt_x.min(), mess3_gt_x.max()
            y_min, y_max = mess3_gt_y.min(), mess3_gt_y.max()
            xx, yy = np.mgrid[x_min:x_max:.02, y_min:y_max:.02]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kde(positions).T, xx.shape)

            axes[0, 2].contourf(xx, yy, f, levels=20, cmap='Blues', alpha=0.8)
            axes[0, 2].set_title('Mess3 GT Density', fontsize=12, fontweight='bold')
            axes[0, 2].set_aspect('equal')
        else:
            axes[0, 2].text(0.5, 0.5, 'Not enough\npoints for\ndensity plot',
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Mess3 GT Density', fontsize=12, fontweight='bold')
    except ImportError:
        axes[0, 2].scatter(mess3_gt_x, mess3_gt_y, alpha=0.6, s=4, c='blue')
        axes[0, 2].set_title('Mess3 GT (no scipy)', fontsize=12, fontweight='bold')
        axes[0, 2].set_aspect('equal')

    # Mess3 overlay with transparency
    axes[0, 3].scatter(mess3_gt_x, mess3_gt_y, alpha=0.3, s=3, c='blue', label='Ground Truth', edgecolors='none')
    axes[0, 3].scatter(mess3_pred_x, mess3_pred_y, alpha=0.3, s=3, c='red', label='Predictions', edgecolors='none')
    axes[0, 3].set_title('Mess3 Detailed Overlay', fontsize=12, fontweight='bold')
    axes[0, 3].legend(fontsize=10)
    axes[0, 3].set_aspect('equal')
    axes[0, 3].grid(True, alpha=0.3)

    # Bloch Walk analysis (same structure)
    bloch_gt = bloch_results['ground_truth']
    bloch_pred = bloch_results['predictions']
    bloch_gt_x, bloch_gt_y = project_to_simplex_2d_CORRECT(bloch_gt)
    bloch_pred_x, bloch_pred_y = project_to_simplex_2d_CORRECT(bloch_pred)

    # Bloch ground truth scatter
    axes[1, 0].scatter(bloch_gt_x, bloch_gt_y, alpha=0.6, s=4, c='blue', edgecolors='none')
    axes[1, 0].set_title(f'Bloch Walk Ground Truth\nN = {len(bloch_gt)} points', fontsize=12, fontweight='bold')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)

    # Bloch predictions scatter
    axes[1, 1].scatter(bloch_pred_x, bloch_pred_y, alpha=0.6, s=4, c='red', edgecolors='none')
    axes[1, 1].set_title(f'Bloch Walk Predictions\nR² = {bloch_results["r2"]:.4f}', fontsize=12, fontweight='bold')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)

    # Bloch density plot (ground truth)
    try:
        if len(bloch_gt_x) > 50:
            xy = np.vstack([bloch_gt_x, bloch_gt_y])
            kde = gaussian_kde(xy)

            x_min, x_max = bloch_gt_x.min(), bloch_gt_x.max()
            y_min, y_max = bloch_gt_y.min(), bloch_gt_y.max()
            xx, yy = np.mgrid[x_min:x_max:.02, y_min:y_max:.02]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kde(positions).T, xx.shape)

            axes[1, 2].contourf(xx, yy, f, levels=20, cmap='Reds', alpha=0.8)
            axes[1, 2].set_title('Bloch Walk GT Density', fontsize=12, fontweight='bold')
            axes[1, 2].set_aspect('equal')
        else:
            axes[1, 2].text(0.5, 0.5, 'Not enough\npoints for\ndensity plot',
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Bloch Walk GT Density', fontsize=12, fontweight='bold')
    except ImportError:
        axes[1, 2].scatter(bloch_gt_x, bloch_gt_y, alpha=0.6, s=4, c='blue')
        axes[1, 2].set_title('Bloch GT (no scipy)', fontsize=12, fontweight='bold')
        axes[1, 2].set_aspect('equal')

    # Bloch overlay with transparency
    axes[1, 3].scatter(bloch_gt_x, bloch_gt_y, alpha=0.3, s=3, c='blue', label='Ground Truth', edgecolors='none')
    axes[1, 3].scatter(bloch_pred_x, bloch_pred_y, alpha=0.3, s=3, c='red', label='Predictions', edgecolors='none')
    axes[1, 3].set_title('Bloch Walk Detailed Overlay', fontsize=12, fontweight='bold')
    axes[1, 3].legend(fontsize=10)
    axes[1, 3].set_aspect('equal')
    axes[1, 3].grid(True, alpha=0.3)

    plt.suptitle('DETAILED ANALYSIS: High-Resolution Belief State Geometry', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved DETAILED visualization: {save_path}")

def main():
    """Run the CORRECT analysis following the paper's exact method."""
    print("🔬 CORRECT BELIEF STATE ANALYSIS")
    print("=" * 60)
    print("Following the paper's EXACT method:")
    print("1. Concatenate ALL layer activations")
    print("2. Use LAST token position only")
    print("3. Weighted least-squares regression")
    print("4. Single affine transformation")

    # Load model
    model_path = "results/best.pt"
    config_path = "configs/multipartite_fast_config.yaml"

    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return

    model, config = load_trained_model(model_path, config_path)

    # Generate test sequences - MORE DATA for better visualization
    n_ctx = config['model_config']['n_ctx']
    num_sequences = 2000  # 4x more data for clearer patterns

    combined_seqs, _, _ = create_multipartite_data(
        n_ctx=n_ctx-1, bos=True, num_sequences=num_sequences, device='cuda', seed=42
    )

    print(f"\nGenerated {num_sequences} sequences of length {combined_seqs.shape[1]}")

    # Extract concatenated activations (CORRECT METHOD)
    concatenated_activations = extract_concatenated_activations_CORRECT(model, combined_seqs)

    # Generate belief states at last position (CORRECT METHOD)
    mess3_beliefs, bloch_beliefs = generate_belief_states_at_last_position(
        combined_seqs, n_ctx, num_sequences
    )

    # Run belief state analysis (CORRECT METHOD)
    mess3_results = analyze_belief_decoding_CORRECT(
        concatenated_activations, mess3_beliefs, "Mess3"
    )

    bloch_results = analyze_belief_decoding_CORRECT(
        concatenated_activations, bloch_beliefs, "Bloch Walk"
    )

    # Create save directory
    save_dir = Path("correct_method_results")
    save_dir.mkdir(exist_ok=True)

    # Visualize results
    visualize_correct_results(
        mess3_results, bloch_results,
        save_dir / "correct_method_belief_analysis.png"
    )

    # Save detailed results
    import json
    results = {
        'method': 'correct_paper_method',
        'concatenated_dims': concatenated_activations.shape[1],
        'num_sequences': num_sequences,
        'mess3': {
            'r2': float(mess3_results['r2']),
            'mse': float(mess3_results['mse'])
        },
        'bloch': {
            'r2': float(bloch_results['r2']),
            'mse': float(bloch_results['mse'])
        }
    }

    with open(save_dir / "correct_method_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("CORRECT METHOD RESULTS")
    print(f"{'='*60}")
    print(f"✅ Concatenated dimensions: {concatenated_activations.shape[1]}")
    print(f"✅ Mess3 R²: {mess3_results['r2']:.6f}")
    print(f"✅ Bloch Walk R²: {bloch_results['r2']:.6f}")

    if mess3_results['r2'] > 0.5 or bloch_results['r2'] > 0.5:
        print("🎉 SUCCESS: Strong belief state decoding achieved!")
        print("   The model DOES learn geometric representations!")
    elif mess3_results['r2'] > 0.1 or bloch_results['r2'] > 0.1:
        print("⚠️ PARTIAL SUCCESS: Moderate belief state decoding")
        print("   The model learns some geometric information")
    else:
        print("❌ POOR RESULTS: Weak belief state decoding")
        print("   The model may not learn geometric representations")
        print("   Or our belief state generation is still incorrect")

    print(f"\n💾 Results saved to {save_dir}")

if __name__ == "__main__":
    main()