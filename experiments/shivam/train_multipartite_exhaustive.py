"""
Train transformer on multipartite process: Mess3 ⊗ Bloch Walk (Exhaustive Generation)

Follows the notebook approach:
- Exhaustive sequence generation using generate_all_seqs()
- Cartesian product with joint probabilities
- BatchGenerator for sampling during training
"""
import torch
import numpy as np
from pathlib import Path
import json
from tqdm.auto import tqdm

from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import mess3, tom_quantum
from epsilon_transformers.training.dataloader import generate_all_seqs, BatchGenerator
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch.nn.functional as F


def create_multipartite_sequences_exhaustive(n_ctx, bos=False, device='cuda'):
    """
    Generate multipartite sequences using EXHAUSTIVE generation.

    Following notebook approach:
    1. Generate ALL Mess3 sequences
    2. Generate ALL Bloch Walk sequences
    3. Create Cartesian product
    4. Compute joint probabilities

    Returns data suitable for BatchGenerator.
    """
    print("="*60)
    print("GENERATING MULTIPARTITE SEQUENCES (EXHAUSTIVE)")
    print("="*60)

    # Create processes
    T_mess3 = mess3(x=0.15, a=0.6)
    T_bloch = tom_quantum(alpha=2.5, beta=0.3)

    ghmm_mess3 = TransitionMatrixGHMM(T_mess3)
    ghmm_mess3.name = "Mess3"

    ghmm_bloch = TransitionMatrixGHMM(T_bloch)
    ghmm_bloch.name = "Bloch Walk"

    print(f"\nMess3 Process:")
    print(f"  Vocabulary: {T_mess3.shape[0]} tokens")
    print(f"  States: {T_mess3.shape[1]}")

    print(f"\nBloch Walk Process:")
    print(f"  Vocabulary: {T_bloch.shape[0]} tokens")
    print(f"  States: {T_bloch.shape[1]}")

    # Generate ALL possible sequences for each process (exhaustive)
    print(f"\nGenerating ALL sequences for each process (n_ctx={n_ctx})...")
    mess3_seqs, mess3_probs, mess3_lb = generate_all_seqs(ghmm_mess3, n_ctx + 1, bos=bos)
    bloch_seqs, bloch_probs, bloch_lb = generate_all_seqs(ghmm_bloch, n_ctx + 1, bos=bos)

    print(f"\nMess3: {len(mess3_seqs):,} unique sequences")
    print(f"Bloch Walk: {len(bloch_seqs):,} unique sequences")

    # Create Cartesian product
    n_mess3 = len(mess3_seqs)
    n_bloch = len(bloch_seqs)
    n_total = n_mess3 * n_bloch

    print(f"\nCreating Cartesian product: {n_mess3:,} × {n_bloch:,} = {n_total:,} sequences")

    # Preallocate arrays
    multipartite_seqs = torch.zeros((n_total, n_ctx + 1), dtype=torch.int32)
    multipartite_probs = torch.zeros(n_total, dtype=torch.float32)

    print("Building Cartesian product...")
    idx = 0
    for i in tqdm(range(n_mess3), desc="Mess3 sequences"):
        for j in range(n_bloch):
            # Combine: token = mess3_token * 4 + bloch_token
            multipartite_seqs[idx] = mess3_seqs[i] * 4 + bloch_seqs[j]
            # Independent processes: P(mess3, bloch) = P(mess3) * P(bloch)
            multipartite_probs[idx] = mess3_probs[i] * bloch_probs[j]
            idx += 1

    # Verify probabilities sum to 1
    prob_sum = multipartite_probs.sum().item()
    print(f"\nProbability sum: {prob_sum:.6f} (should be ≈1.0)")
    if abs(prob_sum - 1.0) > 1e-4:
        print(f"Warning: Probabilities don't sum exactly to 1, normalizing...")
        multipartite_probs = multipartite_probs / prob_sum

    # Compute loss lower bound for multipartite process
    # Since independent: H(X,Y) = H(X) + H(Y)
    loss_lower_bound = mess3_lb + bloch_lb

    print(f"\nMultipartite sequences created:")
    print(f"  Total sequences: {len(multipartite_seqs):,}")
    print(f"  Sequence length: {multipartite_seqs.shape[1]}")
    print(f"  Vocabulary size: {len(torch.unique(multipartite_seqs))}")
    print(f"  Loss lower bound (mean): {loss_lower_bound.mean().item():.4f}")

    # Save metadata (keep original sequences for analysis)
    metadata = {
        'n_mess3': n_mess3,
        'n_bloch': n_bloch,
        'T_mess3': T_mess3.tolist(),
        'T_bloch': T_bloch.tolist(),
        'mess3_seqs': mess3_seqs.cpu().numpy().tolist(),
        'bloch_seqs': bloch_seqs.cpu().numpy().tolist(),
        'mess3_probs': mess3_probs.cpu().numpy().tolist(),
        'bloch_probs': bloch_probs.cpu().numpy().tolist(),
        'mess3_lb': mess3_lb.cpu().numpy().tolist() if torch.is_tensor(mess3_lb) else mess3_lb.tolist(),
        'bloch_lb': bloch_lb.cpu().numpy().tolist() if torch.is_tensor(bloch_lb) else bloch_lb.tolist(),
    }

    return multipartite_seqs.to(device), multipartite_probs.to(device), torch.tensor(loss_lower_bound).to(device), metadata


def validate_model(model, dataloader, loss_lower_bound, device):
    """Evaluate model on full dataset (following notebook approach)"""
    model.eval()
    with torch.no_grad():
        X, Y, probs = dataloader.validation_data()
        X, Y, probs = X.to(device), Y.to(device), probs.to(device)

        logits = model(X)
        batch_size, seq_length, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = Y.reshape(-1).to(torch.int64)

        # Cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = loss.reshape(batch_size, seq_length)

        # Weight by sequence probabilities
        weighted_loss = loss * probs.unsqueeze(1)
        loss_per_position = weighted_loss.sum(dim=0)

        # Normalize by theoretical optimum
        normalized_loss = loss_per_position / loss_lower_bound

    return loss_per_position, normalized_loss


def main():
    output_dir = Path(__file__).parent / "results_exhaustive"
    output_dir.mkdir(exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_ctx = 4  # Small for computational feasibility (3^5 × 4^5 = ~249K sequences)
    bos = False

    print("="*60)
    print("MULTIPARTITE PROCESS EXPERIMENT (EXHAUSTIVE)")
    print("="*60)
    print(f"Mess3 (3 tokens) ⊗ Bloch Walk (4 tokens) = 12 tokens")
    print(f"Context length: {n_ctx}")
    print(f"Device: {device}")
    print()

    # Generate data (exhaustive)
    transformer_inputs, probs, loss_lower_bound, metadata = create_multipartite_sequences_exhaustive(
        n_ctx, bos, device
    )

    # Create dataloader (optimized for 40GB GPU)
    batch_size = 1024  # Increased from 128 (8x larger)
    batches_per_epoch = 200  # Increased from 50 (4x more batches)

    dataloader = BatchGenerator(
        transformer_inputs,
        probs,
        batches_per_epoch,
        batch_size,
        device
    )

    print(f"\n✓ DataLoader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Tokens per epoch: {dataloader.tokens_per_epoch:,}")
    print()

    # Create model (same as notebook)
    d_vocab = len(torch.unique(transformer_inputs))

    model_config = HookedTransformerConfig(
        n_layers=2,
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        d_vocab=d_vocab,
        n_ctx=n_ctx,
        act_fn="relu",
        normalization_type="LN",
        device=device,
        attn_only=False,
        seed=42,
        dtype=torch.float32
    )

    model = HookedTransformer(model_config)

    print(f"✓ Transformer created:")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Model dimension: {model_config.d_model}")
    print(f"  Heads: {model_config.n_heads}")
    print(f"  Head dimension: {model_config.d_head}")
    print(f"  Vocabulary: {d_vocab}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Train (following notebook approach)
    loss_lower_bound_tensor = torch.from_numpy(
        loss_lower_bound.cpu().numpy() if torch.is_tensor(loss_lower_bound) else loss_lower_bound
    ).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)  # Increased learning rate
    n_epochs = 1000
    val_every = 25  # More frequent validation
    history = {'epoch': [], 'loss': [], 'normalized_loss': []}

    print(f"Training for {n_epochs} epochs...")
    print(f"Initial evaluation:")
    val_loss, norm_loss = validate_model(model, dataloader, loss_lower_bound_tensor, device)
    print(f"  Mean loss: {val_loss.mean().item():.4f}")
    print(f"  Mean normalized: {norm_loss.mean().item():.4f}")
    print()

    with tqdm(range(n_epochs), desc="Training") as pbar:
        for epoch in pbar:
            model.train()
            epoch_losses = []

            for input_sequences, target_sequences in dataloader:
                optimizer.zero_grad()

                # Forward pass
                logits = model(input_sequences)

                # Compute loss
                batch_size_actual, seq_length, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = target_sequences.reshape(-1).to(torch.int64)

                loss = F.cross_entropy(logits_flat, targets_flat)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            # Validation
            if epoch % val_every == 0 or epoch == n_epochs - 1:
                val_loss, norm_loss = validate_model(model, dataloader, loss_lower_bound_tensor, device)
                mean_loss = val_loss.mean().item()
                mean_norm = norm_loss.mean().item()

                history['epoch'].append(epoch)
                history['loss'].append(mean_loss)
                history['normalized_loss'].append(mean_norm)

                pbar.set_postfix({
                    'loss': f'{mean_loss:.4f}',
                    'norm': f'{mean_norm:.4f}'
                })

    print(f"\n✓ Training complete!")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Final normalized loss: {history['normalized_loss'][-1]:.4f}")
    print()

    # Save everything
    print("Saving results...")
    torch.save(model.state_dict(), output_dir / "model.pt")
    torch.save(transformer_inputs.cpu(), output_dir / "sequences.pt")
    torch.save(probs.cpu(), output_dir / "probs.pt")

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    config_dict = model.cfg.to_dict()
    config_dict['dtype'] = str(config_dict['dtype'])
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"✓ Saved to {output_dir}/")
    print()
    print("Next: Run analyze_multipartite_geometry_exhaustive.py to analyze belief state representations")


if __name__ == "__main__":
    main()
