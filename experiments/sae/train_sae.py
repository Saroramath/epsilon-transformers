"""
Train Sparse Autoencoder on Transformer Residual Stream Activations

Load activations from trained multipartite model and train SAE to discover
interpretable features corresponding to belief state geometries.
"""
import torch
import numpy as np
from pathlib import Path
import json
from tqdm.auto import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from sae_model import SparseAutoencoder

from transformer_lens import HookedTransformer, HookedTransformerConfig


def extract_activations(model, sequences, layer_idx, device, batch_size=256):
    """
    Extract activations from a specific layer of the transformer.

    Args:
        model: Trained transformer
        sequences: Input sequences [n_seqs, seq_len+1]
        layer_idx: Which layer to extract from
        device: Device
        batch_size: Batch size for processing

    Returns:
        activations: [n_seqs, seq_len, d_model]
    """
    print(f"\nExtracting activations from layer {layer_idx}...")

    all_activations = []
    n_ctx = model.cfg.n_ctx

    num_batches = (len(sequences) + batch_size - 1) // batch_size

    model.eval()
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))

            # Use first n_ctx tokens
            batch = sequences[start_idx:end_idx, :n_ctx].to(device)

            # Run through model with caching
            _, cache = model.run_with_cache(batch)

            # Extract residual stream after the specified layer
            layer_acts = cache[f'blocks.{layer_idx}.hook_resid_post']
            all_activations.append(layer_acts.cpu())

    activations = torch.cat(all_activations, dim=0)
    print(f"✓ Extracted activations: {activations.shape}")

    return activations


def train_sae(
    activations,
    d_hidden,
    sparsity_coeff,
    n_epochs,
    batch_size,
    learning_rate,
    device,
    val_split=0.2,
    tie_weights=False
):
    """
    Train SAE on activations.

    Args:
        activations: Input activations [n_seqs, seq_len, d_model]
        d_hidden: SAE hidden dimension
        sparsity_coeff: L1 sparsity coefficient
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device
        val_split: Fraction of data for validation
        tie_weights: Whether to tie encoder/decoder weights

    Returns:
        sae: Trained SAE model
        history: Training history
    """
    # Flatten activations: [n_seqs * seq_len, d_model]
    n_seqs, seq_len, d_model = activations.shape
    activations_flat = activations.reshape(-1, d_model)

    print(f"\nFlattened activations: {activations_flat.shape}")

    # Train/val split
    n_samples = len(activations_flat)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = activations_flat[train_indices].to(device)
    val_data = activations_flat[val_indices].to(device)

    print(f"Train: {len(train_data):,} samples")
    print(f"Val: {len(val_data):,} samples")

    # Create SAE
    sae = SparseAutoencoder(
        d_model=d_model,
        d_hidden=d_hidden,
        tie_weights=tie_weights,
        device=device
    )

    print(f"\n✓ Created SAE:")
    print(f"  Input dim: {d_model}")
    print(f"  Hidden dim: {d_hidden} ({d_hidden/d_model:.1f}x expansion)")
    print(f"  Tied weights: {tie_weights}")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_recon': [],
        'train_sparsity': [],
        'val_loss': [],
        'val_recon': [],
        'val_sparsity': [],
    }

    # Training loop
    print(f"Training for {n_epochs} epochs...")
    print(f"  Sparsity coefficient: {sparsity_coeff}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print()

    n_batches = (n_train + batch_size - 1) // batch_size

    with tqdm(range(n_epochs), desc="Training") as pbar:
        for epoch in pbar:
            # Training
            sae.train()
            epoch_losses = []
            epoch_recons = []
            epoch_sparsities = []

            # Shuffle training data
            perm = torch.randperm(n_train, device=device)
            train_data_shuffled = train_data[perm]

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_train)
                batch = train_data_shuffled[start:end]

                optimizer.zero_grad()

                # Forward pass
                loss, recon_loss, sparsity_loss = sae.loss(batch, sparsity_coeff)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_recons.append(recon_loss.item())
                epoch_sparsities.append(sparsity_loss.item())

            # Validation
            sae.eval()
            with torch.no_grad():
                val_loss, val_recon, val_sparsity = sae.loss(val_data, sparsity_coeff)

            # Record history
            history['epoch'].append(epoch)
            history['train_loss'].append(np.mean(epoch_losses))
            history['train_recon'].append(np.mean(epoch_recons))
            history['train_sparsity'].append(np.mean(epoch_sparsities))
            history['val_loss'].append(val_loss.item())
            history['val_recon'].append(val_recon.item())
            history['val_sparsity'].append(val_sparsity.item())

            pbar.set_postfix({
                'loss': f'{history["val_loss"][-1]:.4f}',
                'recon': f'{history["val_recon"][-1]:.4f}',
                'sparsity': f'{history["val_sparsity"][-1]:.4f}'
            })

    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final reconstruction: {history['val_recon'][-1]:.4f}")
    print(f"  Final sparsity: {history['val_sparsity'][-1]:.4f}")
    print()

    return sae, history


def main():
    # Configuration
    results_dir = Path(__file__).parent.parent / "shivam" / "results_exhaustive"
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # SAE hyperparameters
    layer_to_analyze = 1  # Analyze layer 1 (best from previous analysis)
    d_hidden = 256  # 4x expansion (64 -> 256)
    sparsity_coeff = 0.01  # L1 coefficient
    n_epochs = 200
    batch_size = 512
    learning_rate = 1e-3
    tie_weights = False

    print("="*70)
    print("SPARSE AUTOENCODER TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Layer: {layer_to_analyze}")
    print(f"Hidden dim: {d_hidden}")
    print(f"Sparsity coeff: {sparsity_coeff}")
    print()

    # Load model
    print("Loading trained transformer...")
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

    # Extract activations
    activations = extract_activations(
        model, sequences, layer_to_analyze, device, batch_size=256
    )

    # Train SAE
    sae, history = train_sae(
        activations=activations,
        d_hidden=d_hidden,
        sparsity_coeff=sparsity_coeff,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        val_split=0.2,
        tie_weights=tie_weights
    )

    # Get feature statistics
    print("Computing feature statistics...")
    activations_flat = activations.reshape(-1, activations.shape[-1]).to(device)
    feature_stats = sae.get_feature_stats(activations_flat)

    print(f"\nFeature Statistics:")
    print(f"  Overall sparsity: {feature_stats['sparsity']:.4f}")
    print(f"  Active features (>1% freq): {(feature_stats['activation_freq'] > 0.01).sum()}")
    print(f"  Active features (>5% freq): {(feature_stats['activation_freq'] > 0.05).sum()}")
    print()

    # Save everything
    print("Saving results...")

    # Save SAE model
    torch.save(sae.state_dict(), output_dir / "sae.pt")

    # Save config
    sae_config = {
        'd_model': sae.d_model,
        'd_hidden': sae.d_hidden,
        'tie_weights': sae.tie_weights,
        'sparsity_coeff': sparsity_coeff,
        'layer_analyzed': layer_to_analyze,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }

    with open(output_dir / "sae_config.json", 'w') as f:
        json.dump(sae_config, f, indent=2)

    # Save training history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Save feature statistics
    feature_stats_serializable = {
        'sparsity': float(feature_stats['sparsity']),
        'activation_freq': feature_stats['activation_freq'].tolist(),
        'mean_activation': feature_stats['mean_activation'].tolist(),
        'max_activation': feature_stats['max_activation'].tolist(),
    }

    with open(output_dir / "feature_stats.json", 'w') as f:
        json.dump(feature_stats_serializable, f, indent=2)

    # Save weights
    np.save(output_dir / "encoder_weights.npy", sae.get_encoder_weights())
    np.save(output_dir / "decoder_weights.npy", sae.get_decoder_weights())

    print(f"✓ Saved to {output_dir}/")
    print()
    print("Next: Run analyze_sae.py to analyze learned features")


if __name__ == "__main__":
    main()
