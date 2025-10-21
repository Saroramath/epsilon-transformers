"""
Train transformer on Bloch Walk only (control experiment)

Tests if a transformer can learn Bloch Walk belief states when trained
on Bloch Walk alone (without Mess3 interference).
"""
import torch
import numpy as np
from pathlib import Path
import json
from tqdm.auto import tqdm

from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import tom_quantum
from epsilon_transformers.training.dataloader import generate_all_seqs, BatchGenerator
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch.nn.functional as F


def main():
    output_dir = Path(__file__).parent / "results_bloch_only"
    output_dir.mkdir(exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_ctx = 4
    bos = False

    print("="*60)
    print("BLOCH WALK ONLY EXPERIMENT")
    print("="*60)
    print(f"Context length: {n_ctx}")
    print(f"Device: {device}")
    print()

    # Create Bloch Walk process
    T_bloch = tom_quantum(alpha=2.5, beta=0.3)
    ghmm_bloch = TransitionMatrixGHMM(T_bloch)
    ghmm_bloch.name = "Bloch Walk"

    print(f"Bloch Walk Process:")
    print(f"  Vocabulary: {T_bloch.shape[0]} tokens")
    print(f"  States: {T_bloch.shape[1]}")
    print()

    # Generate ALL sequences exhaustively
    print(f"Generating ALL Bloch Walk sequences (n_ctx={n_ctx})...")
    sequences, probs, loss_lower_bound = generate_all_seqs(ghmm_bloch, n_ctx + 1, bos=bos)

    print(f"✓ Generated {len(sequences):,} unique sequences")
    print(f"  Loss lower bound (mean): {loss_lower_bound.mean().item():.4f}")
    print()

    # Create dataloader
    batch_size = 1024
    batches_per_epoch = 200

    dataloader = BatchGenerator(
        sequences,
        probs,
        batches_per_epoch,
        batch_size,
        device
    )

    print(f"✓ DataLoader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print()

    # Create model (same architecture as multipartite for comparison)
    d_vocab = T_bloch.shape[0]

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
    print(f"  Vocabulary: {d_vocab}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Train
    loss_lower_bound_tensor = torch.from_numpy(
        loss_lower_bound.cpu().numpy() if torch.is_tensor(loss_lower_bound) else loss_lower_bound
    ).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    n_epochs = 1000
    val_every = 25
    history = {'epoch': [], 'loss': [], 'normalized_loss': []}

    print(f"Training for {n_epochs} epochs...")

    # Initial evaluation
    model.eval()
    with torch.no_grad():
        X, Y, probs_val = dataloader.validation_data()
        X, Y, probs_val = X.to(device), Y.to(device), probs_val.to(device)
        logits = model(X)
        batch_size_val, seq_length, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = Y.reshape(-1).to(torch.int64)
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = loss.reshape(batch_size_val, seq_length)
        weighted_loss = loss * probs_val.unsqueeze(1)
        loss_per_position = weighted_loss.sum(dim=0)
        normalized_loss = loss_per_position / loss_lower_bound_tensor

    print(f"Initial evaluation:")
    print(f"  Mean loss: {loss_per_position.mean().item():.4f}")
    print(f"  Mean normalized: {normalized_loss.mean().item():.4f}")
    print()

    with tqdm(range(n_epochs), desc="Training") as pbar:
        for epoch in pbar:
            model.train()
            epoch_losses = []

            for input_sequences, target_sequences in dataloader:
                optimizer.zero_grad()

                logits = model(input_sequences)
                batch_size_actual, seq_length, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = target_sequences.reshape(-1).to(torch.int64)

                loss = F.cross_entropy(logits_flat, targets_flat)

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            # Validation
            if epoch % val_every == 0 or epoch == n_epochs - 1:
                model.eval()
                with torch.no_grad():
                    X, Y, probs_val = dataloader.validation_data()
                    X, Y, probs_val = X.to(device), Y.to(device), probs_val.to(device)
                    logits = model(X)
                    batch_size_val, seq_length, vocab_size = logits.shape
                    logits_flat = logits.reshape(-1, vocab_size)
                    targets_flat = Y.reshape(-1).to(torch.int64)
                    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                    loss = loss.reshape(batch_size_val, seq_length)
                    weighted_loss = loss * probs_val.unsqueeze(1)
                    loss_per_position = weighted_loss.sum(dim=0)
                    normalized_loss = loss_per_position / loss_lower_bound_tensor

                mean_loss = loss_per_position.mean().item()
                mean_norm = normalized_loss.mean().item()

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
    torch.save(sequences.cpu(), output_dir / "sequences.pt")
    torch.save(probs.cpu(), output_dir / "probs.pt")

    metadata = {
        'n_samples': len(sequences),
        'T_bloch': T_bloch.tolist(),
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    config_dict = model.cfg.to_dict()
    config_dict['dtype'] = str(config_dict['dtype'])
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"✓ Saved to {output_dir}/")


if __name__ == "__main__":
    main()
