"""
Analyze Individual Process (Mess3 or Bloch Walk)

Generic analysis script that works for either process trained individually.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_nn_inputs


def project_to_simplex_2d(beliefs):
    """Project 3D belief states to 2D using barycentric coordinates."""
    x = beliefs[:, 0] - beliefs[:, 1] / 2 - beliefs[:, 2] / 2
    y = np.sqrt(3) / 2 * (beliefs[:, 1] - beliefs[:, 2])

    # Rotate 90 degrees counterclockwise
    x_rot = -y
    y_rot = x

    return x_rot, y_rot


def main(process_name):
    """
    Analyze a single process.

    Args:
        process_name: 'mess3' or 'bloch'
    """
    results_dir = Path(__file__).parent / f"results_{process_name}_only"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print(f"{process_name.upper()} ONLY - GEOMETRY ANALYSIS")
    print("="*70)
    print()

    # Load model
    print("Loading model...")
    with open(results_dir / "config.json", 'r') as f:
        config_dict = json.load(f)

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

    T_matrix = np.array(metadata[f'T_{process_name}'])

    # Compute belief states
    print("Computing belief states using MSP trees...")
    ghmm = TransitionMatrixGHMM(T_matrix)
    n_ctx = model_config.n_ctx
    msp = ghmm.derive_mixed_state_tree(depth=n_ctx+1)

    # Build belief mappings
    msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in msp.belief_states]
    belief_index = {tuple(b): i for i, b in enumerate(set(msp_beliefs))}
    probs_dict = {tuple(path): prob for path, prob in zip(msp.paths, msp.path_probs)}

    print(f"  Unique belief states: {len(belief_index)}")

    # Get beliefs
    print("\nMapping sequences to belief states...")
    beliefs, _, _, _ = get_beliefs_for_nn_inputs(
        sequences.cpu(),
        belief_index,
        msp.paths,
        msp.belief_states,
        msp.unnorm_belief_states,
        probs_dict
    )

    print(f"✓ Beliefs: {beliefs.shape}")
    print()

    # Extract activations
    print("Extracting activations from transformer...")

    all_activations = {f'layer_{i}': [] for i in range(model_config.n_layers)}

    batch_size = 256
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))

            batch = sequences[start_idx:end_idx, :-1].to(device)

            logits, cache = model.run_with_cache(batch)

            for layer_idx in range(model_config.n_layers):
                layer_acts = cache[f'blocks.{layer_idx}.hook_resid_post']
                all_activations[f'layer_{layer_idx}'].append(layer_acts.cpu())

    for key in all_activations:
        all_activations[key] = torch.cat(all_activations[key], dim=0)

    print(f"✓ Extracted activations for {len(sequences):,} sequences")
    print()

    # Perform regression
    print("="*70)
    print("LINEAR REGRESSION: Activations → Belief States")
    print("="*70)
    print()

    results = {}

    for layer_name, acts in all_activations.items():
        print(f"Analyzing {layer_name}...")

        X = acts.reshape(-1, acts.shape[-1]).numpy()
        y = beliefs[:, 1:n_ctx+1, :].reshape(-1, beliefs.shape[-1]).numpy()

        # Sample for speed
        n_samples = min(50000, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]

        # Train regression
        reg = Ridge(alpha=0.1)
        reg.fit(X_sample, y_sample)

        # Predict on full dataset
        y_pred = reg.predict(X)

        # Compute R² per dimension
        r2_per_dim = [r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]
        r2_mean = np.mean(r2_per_dim)

        print(f"  R² per dim: {[f'{r:.3f}' for r in r2_per_dim]}")
        print(f"  Mean R²: {r2_mean:.4f}")
        print()

        results[layer_name] = {
            'r2': r2_mean,
            'r2_per_dim': r2_per_dim,
            'y_pred': y_pred,
        }

    # Visualize best layer
    best_layer = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"Visualizing {best_layer} (highest R²)")
    print()

    # Get predictions for best layer
    y_true = beliefs[:, 1:n_ctx+1, :].reshape(-1, beliefs.shape[-1]).numpy()
    y_pred = results[best_layer]['y_pred']

    # Sample for visualization
    n_vis = min(3000, len(y_true))
    indices = np.random.choice(len(y_true), n_vis, replace=False)

    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x_true, y_true_proj = project_to_simplex_2d(y_true[indices])
    axes[0].scatter(x_true, y_true_proj, alpha=0.3, s=1, c='blue')
    axes[0].set_title(f'{process_name.capitalize()}: Ground Truth\n(Simplex Projection)')
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Barycentric X')
    axes[0].set_ylabel('Barycentric Y')

    x_pred, y_pred_proj = project_to_simplex_2d(y_pred[indices])
    axes[1].scatter(x_pred, y_pred_proj, alpha=0.3, s=1, c='red')
    axes[1].set_title(f'{process_name.capitalize()}: Predicted\n(R² = {results[best_layer]["r2"]:.3f})')
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('Barycentric X')
    axes[1].set_ylabel('Barycentric Y')

    axes[2].scatter(x_true, y_true_proj, alpha=0.2, s=1, c='blue', label='True')
    axes[2].scatter(x_pred, y_pred_proj, alpha=0.2, s=1, c='red', label='Pred')
    axes[2].set_title(f'{process_name.capitalize()}: Overlay')
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('Barycentric X')
    axes[2].set_ylabel('Barycentric Y')
    axes[2].legend()

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
    print(f"Prediction was ~3 dimensions (3 states)")
    print()

    # Save results
    results_to_save = {
        layer: {
            'r2': float(res['r2']),
            'r2_per_dim': [float(r) for r in res['r2_per_dim']],
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
    if len(sys.argv) != 2 or sys.argv[1] not in ['mess3', 'bloch']:
        print("Usage: python analyze_individual_process.py [mess3|bloch]")
        sys.exit(1)

    main(sys.argv[1])
