#!/usr/bin/env python3
"""
H100-Optimized Training script for Multipartite Mess3 × Bloch Walk Transformer

Optimizations included:
- torch.compile for H100 speed
- Mixed precision training
- Larger batch sizes
- Aggressive scheduling for faster convergence
"""

import sys
import os
import torch
import numpy as np
import copy
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from torch.nn import functional as F

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer, HookedTransformerConfig
from epsilon_transformers.training.logger import StructuredLogger
from epsilon_transformers.training.dataloader import get_dataloader_from_data
from multipartite_data import create_multipartite_data

import wandb

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_model_config(logger, model):
    """Save model configuration to JSON."""
    hooked_model_config_dict = copy.deepcopy(model.cfg.to_dict())
    hooked_model_config_dict['dtype'] = str(hooked_model_config_dict['dtype'])
    with open(os.path.join(logger.base_dir, 'hooked_model_config.json'), 'w') as f:
        json.dump(hooked_model_config_dict, f, indent=4)

def train_epoch(model, optimizer, dataset, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    epoch_losses = []

    for input_sequences, target_sequences in dataset:
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_sequences)
                # Reshape for loss calculation
                batch_size, seq_length, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = target_sequences.reshape(-1).to(torch.int64)
                # Compute loss
                loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                loss = loss.reshape(batch_size, seq_length)

            # Backward pass with scaling
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision
            logits = model(input_sequences)
            batch_size, seq_length, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = target_sequences.reshape(-1).to(torch.int64)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = loss.reshape(batch_size, seq_length)
            loss.mean().backward()
            optimizer.step()

        epoch_losses.append(loss.detach())

    return torch.concat(epoch_losses).mean(dim=0)

def validate_epoch(model, dataset, scaler=None):
    """Validate for one epoch."""
    model.eval()

    with torch.no_grad():
        all_losses = []
        for input_sequences, target_sequences in dataset:
            if scaler is not None:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(input_sequences)
            else:
                logits = model(input_sequences)

            batch_size, seq_length, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = target_sequences.reshape(-1).to(torch.int64)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss = loss.reshape(batch_size, seq_length)
            all_losses.append(loss)

        return torch.concat(all_losses).mean(dim=0)

def main():
    """Main training function with H100 optimizations."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Multipartite Transformer (H100 Optimized)')
    parser.add_argument('--config', type=str, default='multipartite_fast_config.yaml',
                       help='Config file name (in configs/ directory)')
    parser.add_argument('--num_sequences', type=int, default=50000,
                       help='Number of training sequences to generate')
    parser.add_argument('--compile_model', action='store_true', default=True,
                       help='Use torch.compile for speed (default: True)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision (default: True)')

    args = parser.parse_args()

    print("🚀 Starting H100-Optimized Multipartite Training")
    print("=" * 60)

    # Load configuration
    config_path = Path(__file__).parent / "configs" / args.config
    config = load_config(config_path)

    print(f"📄 Using config: {args.config}")
    print(f"📊 Generating {args.num_sequences} training sequences")
    print(f"⚡ torch.compile: {'Enabled' if args.compile_model else 'Disabled'}")
    print(f"🎯 Mixed precision: {'Enabled' if args.use_amp else 'Disabled'}")

    # Set up logging
    logger = StructuredLogger(config['experiment_dir'])
    set_seed(42)

    # Initialize Weights & Biases
    if config['global_config']['wandb']:
        wandb.init(
            project=config['global_config']['wandb_project'],
            name=config['run_id'],
            config=config
        )
        print("✅ Weights & Biases initialized")

    # Set device and optimization settings
    device = config['global_config']['device']
    print(f"✅ Using device: {device}")

    # Enable H100 optimizations
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled for H100 performance")

    # Generate multipartite training data
    print("\n📊 Generating multipartite training data...")
    combined_seqs, combined_probs, d_vocab = create_multipartite_data(
        n_ctx=config['model_config']['n_ctx'],
        bos=config['train_config']['bos'],
        num_sequences=args.num_sequences,
        device=device,
        seed=42
    )

    # Create dataloader
    print("🔄 Creating dataloader...")
    dataloader, _ = get_dataloader_from_data(
        combined_seqs,
        combined_probs,
        config['train_config']['batches_per_epoch'],
        config['train_config']['batch_size'],
        device
    )

    # Update vocab size in config
    config['model_config']['d_vocab'] = d_vocab
    config['model_config']['device'] = device
    config['model_config']['dtype'] = getattr(torch, config['model_config']['dtype'])

    # Create model
    print("🤖 Creating transformer model...")
    print(f"   - Layers: {config['model_config']['n_layers']}")
    print(f"   - Heads: {config['model_config']['n_heads']}")
    print(f"   - Head dim: {config['model_config']['d_head']}")
    print(f"   - Model dim: {config['model_config']['d_model']}")
    print(f"   - MLP dim: {config['model_config']['d_mlp']}")
    print(f"   - Vocab size: {d_vocab}")
    print(f"   - Data type: {config['model_config']['dtype']}")

    hooked_model_config = HookedTransformerConfig(**config['model_config'])
    model = HookedTransformer(hooked_model_config)

    # Compile model for H100 speed
    if args.compile_model:
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")
        print("✅ Model compiled for H100 optimization")

    logger.log({"status": "model loaded"})
    save_model_config(logger, model)

    # Create optimizer with better settings
    optimizer = torch.optim.AdamW(  # AdamW typically better than Adam
        model.parameters(),
        lr=config['train_config']['learning_rate'],
        weight_decay=0.01,  # Add weight decay for better generalization
        betas=(0.9, 0.95)   # Better beta2 for stability
    )

    # Create more aggressive scheduler for faster convergence
    scheduler = None
    if config['global_config']['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=200,  # More aggressive
            cooldown=50, threshold=1e-5
        )

    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    print(f"✅ Model created on device: {next(model.parameters()).device}")

    # Training setup
    n_epochs = config['train_config']['n_epochs']
    val_every = config['global_config']['val_every']
    save_every = config['global_config']['save_every']

    print(f"\n🏋️ Training setup:")
    print(f"   - Epochs: {n_epochs}")
    print(f"   - Batch size: {config['train_config']['batch_size']}")
    print(f"   - Learning rate: {config['train_config']['learning_rate']}")
    print(f"   - Validation every: {val_every} epochs")
    print(f"   - Save checkpoint every: {save_every} epochs")

    # Create progress bar
    bar = tqdm(range(n_epochs), desc="Training", unit="epoch")

    # Initial validation and checkpoint
    print("\n📊 Running initial validation...")
    val_loss_per_ctx_pos = validate_epoch(model, dataloader, scaler)
    mean_val_loss = val_loss_per_ctx_pos.mean().item()

    # Only use logger if wandb is enabled, otherwise just save checkpoint
    if config['global_config']['wandb']:
        logger.log_epoch(-1, 0, None, val_loss_per_ctx_pos.tolist(),
                         optimizer.param_groups[0]['lr'])
    logger.save_model_checkpoint(model, "0")

    print(f"Initial validation loss: {mean_val_loss:.6f}")

    # Training loop
    print("\n🚀 Starting training loop...")
    num_tokens_seen = 0
    tokens_per_epoch = config['train_config']['batches_per_epoch'] * \
                      config['train_config']['batch_size'] * \
                      config['model_config']['n_ctx']

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 1000  # Early stopping after 1000 epochs without improvement

    for epoch in bar:
        # Training step
        loss_per_ctx_pos = train_epoch(model, optimizer, dataloader, scaler)
        mean_loss = loss_per_ctx_pos.mean().item()

        # Validation step
        if val_every is not None and epoch % val_every == 0:
            val_loss_per_ctx_pos = validate_epoch(model, dataloader, scaler)
            mean_val_loss = val_loss_per_ctx_pos.mean().item()

            # Early stopping logic
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                patience_counter = 0
                # Save best model
                logger.save_model_checkpoint(model, "best")
            else:
                patience_counter += val_every

            if scheduler:
                scheduler.step(mean_val_loss)

            bar.set_postfix(
                loss=f"{mean_loss:.4f}",
                val_loss=f"{mean_val_loss:.4f}",
                best=f"{best_val_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\n🛑 Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break
        else:
            val_loss_per_ctx_pos = None
            bar.set_postfix(loss=f"{mean_loss:.4f}")

        # Update token count
        num_tokens_seen += tokens_per_epoch

        # Save checkpoint
        if save_every is not None and epoch % save_every == 0 and epoch > 0:
            logger.save_model_checkpoint(model, f"epoch_{epoch}")
            print(f"\n💾 Saved checkpoint at epoch {epoch}")

        # Log metrics
        if config['global_config']['wandb']:
            logger.log_epoch(
                epoch,
                num_tokens_seen,
                loss_per_ctx_pos.tolist(),
                val_loss_per_ctx_pos.tolist() if val_loss_per_ctx_pos is not None else None,
                optimizer.param_groups[0]['lr']
            )

        # Log to wandb
        if config['global_config']['wandb']:
            log_dict = {
                'epoch': epoch,
                'train_loss': mean_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'tokens_seen': num_tokens_seen,
                'best_val_loss': best_val_loss
            }
            if val_loss_per_ctx_pos is not None:
                log_dict['val_loss'] = mean_val_loss
            wandb.log(log_dict)

    # Save final model
    logger.save_model_checkpoint(model, "final")
    print(f"\n✅ Training completed! Final model saved.")
    print(f"📊 Total tokens processed: {num_tokens_seen:,}")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")

    if config['global_config']['wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()