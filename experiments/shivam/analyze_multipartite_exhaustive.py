"""
Analyze Multipartite Geometry (Following Notebook Approach)

Uses proper library functions:
- MSP tree for belief computation
- get_beliefs_for_nn_inputs() for mapping sequences to beliefs
- Ridge regression following notebook pattern
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

from transformer_lens import HookedTransformer, HookedTransformerConfig
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import mess3, tom_quantum
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_nn_inputs


def project_to_simplex_2d(beliefs):
    """
    Project 3D belief states to 2D using barycentric coordinates.
    Critical for visualizing Sierpiński triangle fractals!
    """
    x = beliefs[:, 0] - beliefs[:, 1] / 2 - beliefs[:, 2] / 2
    y = np.sqrt(3) / 2 * (beliefs[:, 1] - beliefs[:, 2])

    # Rotate 90 degrees counterclockwise
    x_rot = -y
    y_rot = x

    return x_rot, y_rot


def compute_orthogonality(W1, W2):
    """Compute orthogonality between two regression weight matrices."""
    # Normalize columns
    W1_norm = W1 / (np.linalg.norm(W1, axis=0, keepdims=True) + 1e-8)
    W2_norm = W2 / (np.linalg.norm(W2, axis=0, keepdims=True) + 1e-8)

    # Compute Frobenius norm of cross-product
    cross_product = W1_norm.T @ W2_norm
    alignment = np.linalg.norm(cross_product, 'fro') / np.sqrt(min(W1.shape[1], W2.shape[1]))

    # Convert to orthogonality score (1 = orthogonal, 0 = aligned)
    orthogonality = 1.0 - alignment

    return orthogonality


def main():
    results_dir = Path(__file__).parent / "results_exhaustive"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("MULTIPARTITE GEOMETRY ANALYSIS (EXHAUSTIVE)")
    print("="*70)
    print()

    # Load model
    print("Loading model...")
    with open(results_dir / "config.json", 'r') as f:
        config_dict = json.load(f)

    # Fix dtype
    if 'dtype' in config_dict and isinstance(config_dict['dtype'], str):
        config_dict['dtype'] = getattr(torch, config_dict['dtype'].split('.')[-1])
    config_dict['device'] = device

    model_config = HookedTransformerConfig(**config_dict)
    model = HookedTransformer(model_config)
    model.load_state_dict(torch.load(results_dir / "model.pt", map_location=device))
    model.eval()

    # Load sequences
    sequences = torch.load(results_dir / "sequences.pt", map_location=device)
    print(f"✓ Loaded model and {len(sequences):,} sequences")
    print()

    # Load metadata
    with open(results_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    T_mess3 = np.array(metadata['T_mess3'])
    T_bloch = np.array(metadata['T_bloch'])
    n_mess3 = metadata['n_mess3']
    n_bloch = metadata['n_bloch']

    # Decode sequences back to component processes
    print("Decoding multipartite sequences...")
    mess3_seqs = sequences // 4
    bloch_seqs = sequences % 4
    print(f"✓ Decoded {len(sequences):,} sequences")
    print(f"  Mess3 component: {mess3_seqs.shape}")
    print(f"  Bloch Walk component: {bloch_seqs.shape}")
    print()

    # Compute belief states following notebook approach
    print("Computing belief states using MSP trees...")

    # Build MSP trees for each process
    ghmm_mess3 = TransitionMatrixGHMM(T_mess3)
    ghmm_bloch = TransitionMatrixGHMM(T_bloch)

    n_ctx = model_config.n_ctx
    msp_mess3 = ghmm_mess3.derive_mixed_state_tree(depth=n_ctx+1)
    msp_bloch = ghmm_bloch.derive_mixed_state_tree(depth=n_ctx+1)

    # Build belief mappings for Mess3
    mess3_msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in msp_mess3.belief_states]
    mess3_belief_index = {tuple(b): i for i, b in enumerate(set(mess3_msp_beliefs))}
    mess3_probs_dict = {tuple(path): prob for path, prob in zip(msp_mess3.paths, msp_mess3.path_probs)}

    # Build belief mappings for Bloch Walk
    bloch_msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in msp_bloch.belief_states]
    bloch_belief_index = {tuple(b): i for i, b in enumerate(set(bloch_msp_beliefs))}
    bloch_probs_dict = {tuple(path): prob for path, prob in zip(msp_bloch.paths, msp_bloch.path_probs)}

    print(f"  Mess3 unique belief states: {len(mess3_belief_index)}")
    print(f"  Bloch Walk unique belief states: {len(bloch_belief_index)}")

    # Get beliefs using library function
    print("\nMapping sequences to belief states...")
    mess3_beliefs, _, _, _ = get_beliefs_for_nn_inputs(
        mess3_seqs.cpu(),
        mess3_belief_index,
        msp_mess3.paths,
        msp_mess3.belief_states,
        msp_mess3.unnorm_belief_states,
        mess3_probs_dict
    )

    bloch_beliefs, _, _, _ = get_beliefs_for_nn_inputs(
        bloch_seqs.cpu(),
        bloch_belief_index,
        msp_bloch.paths,
        msp_bloch.belief_states,
        msp_bloch.unnorm_belief_states,
        bloch_probs_dict
    )

    print(f"✓ Mess3 beliefs: {mess3_beliefs.shape}")
    print(f"✓ Bloch Walk beliefs: {bloch_beliefs.shape}")
    print()

    # Extract activations following notebook approach
    print("Extracting activations from transformer...")

    all_activations = {f'layer_{i}': [] for i in range(model_config.n_layers)}

    batch_size = 256
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))

            # Exclude last token (model expects n_ctx tokens, we have n_ctx+1)
            batch = sequences[start_idx:end_idx, :-1].to(device)

            # Run through model with caching
            logits, cache = model.run_with_cache(batch)

            # Extract residual stream after each layer
            for layer_idx in range(model_config.n_layers):
                layer_acts = cache[f'blocks.{layer_idx}.hook_resid_post']
                all_activations[f'layer_{layer_idx}'].append(layer_acts.cpu())

    # Concatenate batches
    for key in all_activations:
        all_activations[key] = torch.cat(all_activations[key], dim=0)
        print(f"  {key}: {all_activations[key].shape}")

    print(f"✓ Extracted activations for {len(sequences):,} sequences")
    print()

    # Perform regression following notebook approach
    print("="*70)
    print("LINEAR REGRESSION: Activations → Belief States")
    print("="*70)
    print()

    results = {}

    for layer_name, acts in all_activations.items():
        print(f"Analyzing {layer_name}...")

        # Reshape: [num_sequences, seq_len, d_model] → [num_sequences * seq_len, d_model]
        # Activations at positions 0:n_ctx correspond to beliefs at positions 1:n_ctx+1
        X = acts.reshape(-1, acts.shape[-1]).numpy()

        # Mess3 beliefs (positions 1:n_ctx+1)
        y_mess3 = mess3_beliefs[:, 1:n_ctx+1, :].reshape(-1, mess3_beliefs.shape[-1]).numpy()

        # Bloch Walk beliefs (positions 1:n_ctx+1)
        y_bloch = bloch_beliefs[:, 1:n_ctx+1, :].reshape(-1, bloch_beliefs.shape[-1]).numpy()

        # Sample for speed
        n_samples = min(50000, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
        y_mess3_sample = y_mess3[indices]
        y_bloch_sample = y_bloch[indices]

        # Train regressions
        reg_mess3 = Ridge(alpha=0.1)
        reg_mess3.fit(X_sample, y_mess3_sample)

        reg_bloch = Ridge(alpha=0.1)
        reg_bloch.fit(X_sample, y_bloch_sample)

        # Predict on full dataset
        y_mess3_pred = reg_mess3.predict(X)
        y_bloch_pred = reg_bloch.predict(X)

        # Compute R² per dimension
        r2_mess3 = [r2_score(y_mess3[:, i], y_mess3_pred[:, i]) for i in range(y_mess3.shape[1])]
        r2_bloch = [r2_score(y_bloch[:, i], y_bloch_pred[:, i]) for i in range(y_bloch.shape[1])]

        r2_mess3_mean = np.mean(r2_mess3)
        r2_bloch_mean = np.mean(r2_bloch)

        # Compute orthogonality
        orthogonality = compute_orthogonality(reg_mess3.coef_.T, reg_bloch.coef_.T)

        print(f"  Mess3 R² per dim: {[f'{r:.3f}' for r in r2_mess3]}")
        print(f"  Mess3 mean R²: {r2_mess3_mean:.4f}")
        print(f"  Bloch Walk R² per dim: {[f'{r:.3f}' for r in r2_bloch]}")
        print(f"  Bloch Walk mean R²: {r2_bloch_mean:.4f}")
        print(f"  Orthogonality: {orthogonality:.4f}")
        print()

        results[layer_name] = {
            'r2_mess3': r2_mess3_mean,
            'r2_bloch': r2_bloch_mean,
            'orthogonality': orthogonality,
            'y_mess3_pred': y_mess3_pred,
            'y_bloch_pred': y_bloch_pred,
        }

    # Visualize best layer
    best_layer = max(results.items(), key=lambda x: x[1]['r2_mess3'])[0]
    print(f"Visualizing {best_layer} (highest Mess3 R²)")
    print()

    # Get predictions for best layer
    acts = all_activations[best_layer].reshape(-1, all_activations[best_layer].shape[-1]).numpy()
    y_mess3_true = mess3_beliefs[:, 1:n_ctx+1, :].reshape(-1, mess3_beliefs.shape[-1]).numpy()
    y_bloch_true = bloch_beliefs[:, 1:n_ctx+1, :].reshape(-1, bloch_beliefs.shape[-1]).numpy()
    y_mess3_pred = results[best_layer]['y_mess3_pred']
    y_bloch_pred = results[best_layer]['y_bloch_pred']

    # Sample for visualization
    n_vis = 3000
    indices = np.random.choice(len(acts), n_vis, replace=False)

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Mess3
    x_true, y_true = project_to_simplex_2d(y_mess3_true[indices])
    axes[0, 0].scatter(x_true, y_true, alpha=0.3, s=1, c='blue')
    axes[0, 0].set_title('Mess3: Ground Truth\n(Simplex Projection)')
    axes[0, 0].set_aspect('equal')

    x_pred, y_pred = project_to_simplex_2d(y_mess3_pred[indices])
    axes[0, 1].scatter(x_pred, y_pred, alpha=0.3, s=1, c='red')
    axes[0, 1].set_title(f'Mess3: Predicted\n(R² = {results[best_layer]["r2_mess3"]:.3f})')
    axes[0, 1].set_aspect('equal')

    axes[0, 2].scatter(x_true, y_true, alpha=0.2, s=1, c='blue', label='True')
    axes[0, 2].scatter(x_pred, y_pred, alpha=0.2, s=1, c='red', label='Pred')
    axes[0, 2].set_title('Mess3: Overlay')
    axes[0, 2].set_aspect('equal')
    axes[0, 2].legend()

    # Row 2: Bloch Walk (simplex projection - same as Mess3)
    x_true_bloch, y_true_bloch = project_to_simplex_2d(y_bloch_true[indices])
    axes[1, 0].scatter(x_true_bloch, y_true_bloch, alpha=0.3, s=1, c='blue')
    axes[1, 0].set_title('Bloch Walk: Ground Truth\n(Simplex Projection)')
    axes[1, 0].set_xlabel('Barycentric X')
    axes[1, 0].set_ylabel('Barycentric Y')
    axes[1, 0].set_aspect('equal')

    x_pred_bloch, y_pred_bloch = project_to_simplex_2d(y_bloch_pred[indices])
    axes[1, 1].scatter(x_pred_bloch, y_pred_bloch, alpha=0.3, s=1, c='red')
    axes[1, 1].set_title(f'Bloch Walk: Predicted\n(R² = {results[best_layer]["r2_bloch"]:.3f})')
    axes[1, 1].set_xlabel('Barycentric X')
    axes[1, 1].set_ylabel('Barycentric Y')
    axes[1, 1].set_aspect('equal')

    axes[1, 2].scatter(x_true_bloch, y_true_bloch, alpha=0.2, s=1, c='blue', label='True')
    axes[1, 2].scatter(x_pred_bloch, y_pred_bloch, alpha=0.2, s=1, c='red', label='Pred')
    axes[1, 2].set_title('Bloch Walk: Overlay')
    axes[1, 2].set_xlabel('Barycentric X')
    axes[1, 2].set_ylabel('Barycentric Y')
    axes[1, 2].legend()
    axes[1, 2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(results_dir / 'belief_predictions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to belief_predictions.png")
    print()

    # PCA analysis
    print("="*70)
    print("DIMENSIONALITY ANALYSIS")
    print("="*70)
    print()

    acts_all = all_activations['layer_0'].reshape(-1, all_activations['layer_0'].shape[-1]).numpy()
    pca = PCA()
    pca.fit(acts_all)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_dims_90 = np.argmax(cumvar >= 0.90) + 1

    print(f"Effective dimensionality (90% variance): {n_dims_90}")
    print(f"Prediction was ~7 dimensions (3 + 4)")
    print()

    # Save results
    results_to_save = {
        layer: {
            'r2_mess3': float(res['r2_mess3']),
            'r2_bloch': float(res['r2_bloch']),
            'orthogonality': float(res['orthogonality']),
        }
        for layer, res in results.items()
    }
    results_to_save['dimensionality'] = {'n_dims_90': int(n_dims_90)}

    with open(results_dir / 'analysis_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"✓ Saved analysis results")
    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
