"""
Analyze Trained SAE Features

Investigate what features the SAE learned and how they relate to:
1. Mess3 belief states (3D simplex)
2. Bloch Walk belief states (4D simplex)
3. Token structure (12 tokens from Cartesian product)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from sae_model import SparseAutoencoder

from transformer_lens import HookedTransformer, HookedTransformerConfig
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import mess3, tom_quantum
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_nn_inputs


def project_to_simplex_2d(beliefs):
    """
    Project 3D belief states to 2D using barycentric coordinates.
    Used for Mess3.
    """
    x = beliefs[:, 0] - beliefs[:, 1] / 2 - beliefs[:, 2] / 2
    y = np.sqrt(3) / 2 * (beliefs[:, 1] - beliefs[:, 2])
    # Rotate 90 degrees counterclockwise
    x_rot = -y
    y_rot = x
    return x_rot, y_rot


def project_direct_indexing(beliefs, inds=[1, 2]):
    """
    Direct indexing of belief dimensions for visualization.
    Used for TomQA/Bloch Walk - plots dimensions [1, 2] directly.
    Matches Fig2.py approach for TomQA.
    """
    return beliefs[:, inds[0]], beliefs[:, inds[1]]


def transform_for_alpha(weights, min_alpha=0.1, transformation='cbrt'):
    """Transform weights to alpha values for visualization (from Fig2.py)."""
    weights = np.asarray(weights)
    weights = np.clip(weights, 0, 1)

    if transformation == 'log':
        alpha = np.log1p(weights * 100) / np.log1p(100)
    elif transformation == 'sqrt':
        alpha = np.sqrt(weights)
    elif transformation == 'cbrt':
        alpha = np.cbrt(weights)
    elif transformation == 'linear':
        alpha = weights
    else:
        alpha = weights

    alpha = alpha * (1 - min_alpha) + min_alpha
    return alpha


def main():
    results_dir = Path(__file__).parent / "results"
    shivam_results = Path(__file__).parent.parent / "shivam" / "results_exhaustive"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("SAE FEATURE ANALYSIS")
    print("="*70)
    print()

    # Load SAE
    print("Loading trained SAE...")
    with open(results_dir / "sae_config.json", 'r') as f:
        sae_config = json.load(f)

    sae = SparseAutoencoder(
        d_model=sae_config['d_model'],
        d_hidden=sae_config['d_hidden'],
        tie_weights=sae_config['tie_weights'],
        device=device
    )
    sae.load_state_dict(torch.load(results_dir / "sae.pt", map_location=device))
    sae.eval()
    print(f"✓ Loaded SAE: {sae.d_model} → {sae.d_hidden}")
    print()

    # Load transformer and sequences
    print("Loading transformer model...")
    with open(shivam_results / "config.json", 'r') as f:
        config_dict = json.load(f)

    if 'dtype' in config_dict and isinstance(config_dict['dtype'], str):
        config_dict['dtype'] = getattr(torch, config_dict['dtype'].split('.')[-1])
    config_dict['device'] = device

    model_config = HookedTransformerConfig(**config_dict)
    model = HookedTransformer(model_config)
    model.load_state_dict(torch.load(shivam_results / "model.pt", map_location=device))
    model.eval()

    sequences = torch.load(shivam_results / "sequences.pt", map_location=device)
    print(f"✓ Loaded model and {len(sequences):,} sequences")
    print()

    # Load metadata and compute belief states
    print("Computing belief states...")
    with open(shivam_results / "metadata.json", 'r') as f:
        metadata = json.load(f)

    T_mess3 = np.array(metadata['T_mess3'])
    T_bloch = np.array(metadata['T_bloch'])

    # Decode sequences
    mess3_seqs = sequences // 4
    bloch_seqs = sequences % 4

    # Build MSP trees
    ghmm_mess3 = TransitionMatrixGHMM(T_mess3)
    ghmm_bloch = TransitionMatrixGHMM(T_bloch)

    n_ctx = model_config.n_ctx
    msp_mess3 = ghmm_mess3.derive_mixed_state_tree(depth=n_ctx+1)
    msp_bloch = ghmm_bloch.derive_mixed_state_tree(depth=n_ctx+1)

    # Build belief mappings
    mess3_msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in msp_mess3.belief_states]
    mess3_belief_index = {tuple(b): i for i, b in enumerate(set(mess3_msp_beliefs))}
    mess3_probs_dict = {tuple(path): prob for path, prob in zip(msp_mess3.paths, msp_mess3.path_probs)}

    bloch_msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in msp_bloch.belief_states]
    bloch_belief_index = {tuple(b): i for i, b in enumerate(set(bloch_msp_beliefs))}
    bloch_probs_dict = {tuple(path): prob for path, prob in zip(msp_bloch.paths, msp_bloch.path_probs)}

    # Get beliefs and probabilities
    mess3_beliefs, _, _, mess3_probs = get_beliefs_for_nn_inputs(
        mess3_seqs.cpu(),
        mess3_belief_index,
        msp_mess3.paths,
        msp_mess3.belief_states,
        msp_mess3.unnorm_belief_states,
        mess3_probs_dict
    )

    bloch_beliefs, _, _, bloch_probs = get_beliefs_for_nn_inputs(
        bloch_seqs.cpu(),
        bloch_belief_index,
        msp_bloch.paths,
        msp_bloch.belief_states,
        msp_bloch.unnorm_belief_states,
        bloch_probs_dict
    )

    print(f"✓ Computed beliefs: Mess3 {mess3_beliefs.shape}, Bloch {bloch_beliefs.shape}")
    print(f"✓ Computed probabilities: Mess3 {mess3_probs.shape}, Bloch {bloch_probs.shape}")
    print()

    # Extract activations and SAE features
    print("Extracting activations and SAE features...")
    layer_idx = sae_config['layer_analyzed']

    all_activations = []
    all_sae_features = []

    batch_size = 256
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))

            batch = sequences[start_idx:end_idx, :n_ctx].to(device)

            # Get activations
            _, cache = model.run_with_cache(batch)
            layer_acts = cache[f'blocks.{layer_idx}.hook_resid_post']

            # Get SAE features
            acts_flat = layer_acts.reshape(-1, layer_acts.shape[-1])
            _, sae_feats = sae.forward(acts_flat)

            all_activations.append(layer_acts.cpu())
            all_sae_features.append(sae_feats.reshape(layer_acts.shape[0], layer_acts.shape[1], -1).cpu())

    activations = torch.cat(all_activations, dim=0)
    sae_features = torch.cat(all_sae_features, dim=0)

    print(f"✓ Activations: {activations.shape}")
    print(f"✓ SAE features: {sae_features.shape}")
    print()

    # Flatten for analysis
    acts_flat = activations.reshape(-1, activations.shape[-1]).numpy()
    sae_flat = sae_features.reshape(-1, sae_features.shape[-1]).numpy()

    mess3_beliefs_flat = mess3_beliefs[:, 1:n_ctx+1, :].reshape(-1, mess3_beliefs.shape[-1]).numpy()
    bloch_beliefs_flat = bloch_beliefs[:, 1:n_ctx+1, :].reshape(-1, bloch_beliefs.shape[-1]).numpy()

    # Flatten probabilities for alpha weighting
    mess3_probs_flat = np.repeat(mess3_probs.numpy(), n_ctx)
    bloch_probs_flat = np.repeat(bloch_probs.numpy(), n_ctx)

    print(f"Flattened shapes:")
    print(f"  Activations: {acts_flat.shape}")
    print(f"  SAE features: {sae_flat.shape}")
    print(f"  Mess3 beliefs: {mess3_beliefs_flat.shape}")
    print(f"  Bloch beliefs: {bloch_beliefs_flat.shape}")
    print(f"  Mess3 probs: {mess3_probs_flat.shape}")
    print(f"  Bloch probs: {bloch_probs_flat.shape}")
    print()

    # =================================================================
    # ANALYSIS 1: SAE Features → Belief States
    # =================================================================
    print("="*70)
    print("ANALYSIS 1: SAE Features → Belief States")
    print("="*70)
    print()

    # Sample for speed
    n_samples = min(50000, len(sae_flat))
    indices = np.random.choice(len(sae_flat), n_samples, replace=False)

    X_sample = sae_flat[indices]
    y_mess3_sample = mess3_beliefs_flat[indices]
    y_bloch_sample = bloch_beliefs_flat[indices]

    # Regression: SAE features → Mess3 beliefs
    reg_mess3 = Ridge(alpha=0.1)
    reg_mess3.fit(X_sample, y_mess3_sample)

    y_mess3_pred_sae = reg_mess3.predict(sae_flat)
    r2_mess3_sae = [r2_score(mess3_beliefs_flat[:, i], y_mess3_pred_sae[:, i]) for i in range(3)]
    r2_mess3_sae_mean = np.mean(r2_mess3_sae)

    print(f"SAE Features → Mess3 Beliefs:")
    print(f"  R² per dim: {[f'{r:.3f}' for r in r2_mess3_sae]}")
    print(f"  Mean R²: {r2_mess3_sae_mean:.4f}")
    print()

    # Regression: SAE features → Bloch beliefs
    reg_bloch = Ridge(alpha=0.1)
    reg_bloch.fit(X_sample, y_bloch_sample)

    y_bloch_pred_sae = reg_bloch.predict(sae_flat)
    r2_bloch_sae = [r2_score(bloch_beliefs_flat[:, i], y_bloch_pred_sae[:, i]) for i in range(3)]
    r2_bloch_sae_mean = np.mean(r2_bloch_sae)

    print(f"SAE Features → Bloch Beliefs:")
    print(f"  R² per dim: {[f'{r:.3f}' for r in r2_bloch_sae]}")
    print(f"  Mean R²: {r2_bloch_sae_mean:.4f}")
    print()

    # Compare to raw activations
    reg_mess3_acts = Ridge(alpha=0.1)
    reg_mess3_acts.fit(acts_flat[indices], y_mess3_sample)
    y_mess3_pred_acts = reg_mess3_acts.predict(acts_flat)
    r2_mess3_acts = np.mean([r2_score(mess3_beliefs_flat[:, i], y_mess3_pred_acts[:, i]) for i in range(3)])

    reg_bloch_acts = Ridge(alpha=0.1)
    reg_bloch_acts.fit(acts_flat[indices], y_bloch_sample)
    y_bloch_pred_acts = reg_bloch_acts.predict(acts_flat)
    r2_bloch_acts = np.mean([r2_score(bloch_beliefs_flat[:, i], y_bloch_pred_acts[:, i]) for i in range(3)])

    print(f"Comparison (Raw Activations → Beliefs):")
    print(f"  Mess3 R²: {r2_mess3_acts:.4f}")
    print(f"  Bloch R²: {r2_bloch_acts:.4f}")
    print()

    # =================================================================
    # ANALYSIS 2: Per-Feature Belief Correlation
    # =================================================================
    print("="*70)
    print("ANALYSIS 2: Per-Feature Belief Correlation")
    print("="*70)
    print()

    # For each SAE feature, compute correlation with each belief dimension
    mess3_correlations = np.zeros((sae_flat.shape[1], 3))
    bloch_correlations = np.zeros((sae_flat.shape[1], 3))

    for feat_idx in range(sae_flat.shape[1]):
        feat = sae_flat[:, feat_idx]

        for i in range(3):
            mess3_correlations[feat_idx, i] = np.corrcoef(feat, mess3_beliefs_flat[:, i])[0, 1]

        for i in range(3):
            bloch_correlations[feat_idx, i] = np.corrcoef(feat, bloch_beliefs_flat[:, i])[0, 1]

    # Find features most correlated with each process
    max_mess3_corr = np.max(np.abs(mess3_correlations), axis=1)
    max_bloch_corr = np.max(np.abs(bloch_correlations), axis=1)

    # Features strongly correlated with Mess3
    mess3_features = np.where(max_mess3_corr > 0.3)[0]
    print(f"Features correlated with Mess3 (|r| > 0.3): {len(mess3_features)}")
    if len(mess3_features) > 0:
        top_mess3 = mess3_features[np.argsort(max_mess3_corr[mess3_features])[-5:]]
        print(f"  Top 5: {top_mess3} (max |r|: {max_mess3_corr[top_mess3]})")

    # Features strongly correlated with Bloch
    bloch_features = np.where(max_bloch_corr > 0.3)[0]
    print(f"Features correlated with Bloch (|r| > 0.3): {len(bloch_features)}")
    if len(bloch_features) > 0:
        top_bloch = bloch_features[np.argsort(max_bloch_corr[bloch_features])[-5:]]
        print(f"  Top 5: {top_bloch} (max |r|: {max_bloch_corr[top_bloch]})")

    # Features correlated with both (entangled)
    entangled_features = np.where((max_mess3_corr > 0.2) & (max_bloch_corr > 0.2))[0]
    print(f"Entangled features (both |r| > 0.2): {len(entangled_features)}")
    print()

    # =================================================================
    # ANALYSIS 3: Feature Activation Patterns by Token
    # =================================================================
    print("="*70)
    print("ANALYSIS 3: Feature Activation by Token")
    print("="*70)
    print()

    # Get token sequences (flatten)
    tokens_flat = sequences[:, :n_ctx].reshape(-1).cpu().numpy()

    # For top features, compute mean activation per token
    if len(mess3_features) > 0 and len(bloch_features) > 0:
        top_mess3_feat = mess3_features[np.argmax(max_mess3_corr[mess3_features])]
        top_bloch_feat = bloch_features[np.argmax(max_bloch_corr[bloch_features])]

        mess3_by_token = [sae_flat[tokens_flat == t, top_mess3_feat].mean() for t in range(12)]
        bloch_by_token = [sae_flat[tokens_flat == t, top_bloch_feat].mean() for t in range(12)]

        print(f"Top Mess3 feature (#{top_mess3_feat}):")
        print(f"  Activation by token: {[f'{x:.3f}' for x in mess3_by_token]}")
        print()

        print(f"Top Bloch feature (#{top_bloch_feat}):")
        print(f"  Activation by token: {[f'{x:.3f}' for x in bloch_by_token]}")
        print()

    # =================================================================
    # ANALYSIS 4: Visualizations
    # =================================================================
    print("="*70)
    print("ANALYSIS 4: Creating Visualizations")
    print("="*70)
    print()

    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Feature activation frequencies
    ax1 = plt.subplot(2, 4, 1)
    with open(results_dir / "feature_stats.json", 'r') as f:
        feature_stats = json.load(f)
    activation_freq = np.array(feature_stats['activation_freq'])
    ax1.hist(activation_freq, bins=50, edgecolor='black')
    ax1.set_xlabel('Activation Frequency')
    ax1.set_ylabel('Count')
    ax1.set_title('SAE Feature Sparsity Distribution')
    ax1.axvline(0.01, color='red', linestyle='--', label='1% threshold')
    ax1.legend()

    # Plot 2: Max correlation with Mess3
    ax2 = plt.subplot(2, 4, 2)
    ax2.hist(max_mess3_corr, bins=50, edgecolor='black')
    ax2.set_xlabel('Max |Correlation| with Mess3 Beliefs')
    ax2.set_ylabel('Count')
    ax2.set_title('Mess3 Feature Correlations')
    ax2.axvline(0.3, color='red', linestyle='--', label='0.3 threshold')
    ax2.legend()

    # Plot 3: Max correlation with Bloch
    ax3 = plt.subplot(2, 4, 3)
    # Filter out NaN values
    valid_bloch_corr = max_bloch_corr[~np.isnan(max_bloch_corr)]
    ax3.hist(valid_bloch_corr, bins=50, edgecolor='black')
    ax3.set_xlabel('Max |Correlation| with Bloch Beliefs')
    ax3.set_ylabel('Count')
    ax3.set_title('Bloch Feature Correlations')
    ax3.axvline(0.3, color='red', linestyle='--', label='0.3 threshold')
    ax3.legend()

    # Plot 4: 2D scatter of correlations
    ax4 = plt.subplot(2, 4, 4)
    # Filter out NaN values for scatter plot
    valid_indices = ~(np.isnan(max_mess3_corr) | np.isnan(max_bloch_corr))
    ax4.scatter(max_mess3_corr[valid_indices], max_bloch_corr[valid_indices], alpha=0.3, s=10)
    ax4.set_xlabel('Max |r| with Mess3')
    ax4.set_ylabel('Max |r| with Bloch')
    ax4.set_title('Feature Separation: Mess3 vs Bloch')
    ax4.axhline(0.3, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(0.3, color='red', linestyle='--', alpha=0.5)

    # Plot 5-6: Mess3 belief predictions with proper coloring (matching Fig2.py)
    n_vis = 3000
    vis_indices = np.random.choice(len(mess3_beliefs_flat), n_vis, replace=False)

    # Mess3: Use simplex projection
    x_true, y_true = project_to_simplex_2d(mess3_beliefs_flat[vis_indices])
    x_pred, y_pred = project_to_simplex_2d(y_mess3_pred_sae[vis_indices])

    # Compute RGB colors for Mess3 (using ground truth belief dimensions)
    def normalize_dim(data):
        min_val, max_val = np.nanmin(data), np.nanmax(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        return np.ones_like(data) * 0.5

    R_mess3 = normalize_dim(mess3_beliefs_flat[vis_indices, 0])
    G_mess3 = normalize_dim(mess3_beliefs_flat[vis_indices, 1])
    B_mess3 = normalize_dim(mess3_beliefs_flat[vis_indices, 2])

    # Compute alpha from probabilities
    alpha_mess3 = transform_for_alpha(mess3_probs_flat[vis_indices], min_alpha=0.2, transformation='cbrt')
    colors_mess3 = np.stack([R_mess3, G_mess3, B_mess3, alpha_mess3], axis=-1)

    ax5 = plt.subplot(2, 4, 5)
    ax5.scatter(x_true, y_true, color=colors_mess3, s=1, rasterized=True)
    ax5.set_title(f'Mess3: Ground Truth\n(Simplex Projection)', fontsize=10)
    ax5.set_aspect('equal')
    ax5.set_axis_off()

    ax6 = plt.subplot(2, 4, 6)
    ax6.scatter(x_pred, y_pred, color=colors_mess3, s=0.5, rasterized=True)
    ax6.set_title(f'Mess3: SAE Prediction\n(R²={r2_mess3_sae_mean:.3f})', fontsize=10)
    ax6.set_aspect('equal')
    ax6.set_axis_off()

    # Plot 7-8: Bloch belief predictions with direct indexing (like TomQA)
    x_true_bloch, y_true_bloch = project_direct_indexing(bloch_beliefs_flat[vis_indices], inds=[1, 2])
    x_pred_bloch, y_pred_bloch = project_direct_indexing(y_bloch_pred_sae[vis_indices], inds=[1, 2])

    # Compute RGB colors for Bloch using plotted dimensions
    R_bloch = normalize_dim(x_true_bloch)
    G_bloch = normalize_dim(y_true_bloch)
    # Use dimension 0 for blue channel
    B_bloch = normalize_dim(bloch_beliefs_flat[vis_indices, 0])

    # Compute alpha from probabilities
    alpha_bloch = transform_for_alpha(bloch_probs_flat[vis_indices], min_alpha=0.15, transformation='cbrt')
    colors_bloch = np.stack([R_bloch, G_bloch, B_bloch, alpha_bloch], axis=-1)

    ax7 = plt.subplot(2, 4, 7)
    ax7.scatter(x_true_bloch, y_true_bloch, color=colors_bloch, s=0.15, rasterized=True)
    ax7.set_title(f'Bloch: Ground Truth\n(Direct Indexing [1,2])', fontsize=10)
    ax7.set_aspect('equal')
    ax7.set_axis_off()

    ax8 = plt.subplot(2, 4, 8)
    ax8.scatter(x_pred_bloch, y_pred_bloch, color=colors_bloch, s=0.05, rasterized=True)
    ax8.set_title(f'Bloch: SAE Prediction\n(R²={r2_bloch_sae_mean:.3f})', fontsize=10)
    ax8.set_aspect('equal')
    ax8.set_axis_off()

    plt.tight_layout()
    plt.savefig(results_dir / 'sae_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to sae_analysis.png")
    print()

    # Save analysis results
    analysis_results = {
        'sae_to_beliefs': {
            'mess3_r2': float(r2_mess3_sae_mean),
            'bloch_r2': float(r2_bloch_sae_mean),
        },
        'raw_activations_to_beliefs': {
            'mess3_r2': float(r2_mess3_acts),
            'bloch_r2': float(r2_bloch_acts),
        },
        'feature_separation': {
            'n_mess3_features': int(len(mess3_features)),
            'n_bloch_features': int(len(bloch_features)),
            'n_entangled_features': int(len(entangled_features)),
        },
        'sparsity': {
            'overall': float(feature_stats['sparsity']),
            'active_1pct': int((activation_freq > 0.01).sum()),
            'active_5pct': int((activation_freq > 0.05).sum()),
        }
    }

    with open(results_dir / 'analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print("Summary:")
    print(f"  SAE → Mess3: R² = {r2_mess3_sae_mean:.4f}")
    print(f"  SAE → Bloch: R² = {r2_bloch_sae_mean:.4f}")
    print(f"  Mess3 features: {len(mess3_features)}")
    print(f"  Bloch features: {len(bloch_features)}")
    print(f"  Entangled features: {len(entangled_features)}")
    print(f"  Overall sparsity: {feature_stats['sparsity']:.4f}")


if __name__ == "__main__":
    main()
