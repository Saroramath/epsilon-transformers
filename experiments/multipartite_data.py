#!/usr/bin/env python3
"""
Multipartite Data Generation for Mess3 × Bloch Walk

Generate training data by taking Cartesian product of ALL sequences
from Mess3 and Bloch Walk processes using existing infrastructure.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from itertools import product

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from epsilon_transformers.training.dataloader import generate_all_seqs, get_dataloader_from_data, BatchGenerator
from epsilon_transformers.process.transition_matrices import mess3, tom_quantum
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM

def create_multipartite_data(n_ctx=8, bos=True, num_sequences=10000, device='cuda', seed=42):
    """
    Create multipartite training data by independently sampling Mess3 and Bloch Walk sequences
    and pairing them to preserve temporal structure.

    Args:
        n_ctx (int): Context length for sequences
        bos (bool): Whether to include beginning-of-sequence token
        num_sequences (int): Number of independent sequence pairs to generate
        device (str): Device for tensors
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (combined_sequences, combined_probs, d_vocab)
    """
    print("Creating Multipartite Mess3 × Bloch Walk Data")
    print("=" * 50)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create individual processes
    T_mess3 = mess3(a=0.85, x=0.05)
    T_bloch = tom_quantum(alpha=1.0, beta=np.sqrt(51))

    ghmm_mess3 = TransitionMatrixGHMM(T_mess3)
    ghmm_bloch = TransitionMatrixGHMM(T_bloch)

    print(f"Mess3 process: {T_mess3.shape[0]} symbols, {T_mess3.shape[1]} states")
    print(f"Bloch process: {T_bloch.shape[0]} symbols, {T_bloch.shape[1]} states")
    print(f"Generating {num_sequences} independent sequence pairs...")

    # Sample independent sequences from each process
    mess3_sequences = []
    bloch_sequences = []

    print("Sampling Mess3 sequences...")
    for i in range(num_sequences):
        seq = list(ghmm_mess3.yield_emissions(sequence_len=n_ctx))
        mess3_sequences.append(seq)

    print("Sampling Bloch Walk sequences...")
    for i in range(num_sequences):
        seq = list(ghmm_bloch.yield_emissions(sequence_len=n_ctx))
        bloch_sequences.append(seq)

    print(f"Generated {len(mess3_sequences)} Mess3 and {len(bloch_sequences)} Bloch sequences")

    # Create paired sequences with proper tokenization
    combined_seqs = []

    # Vocabulary mapping: (mess3_token, bloch_token) -> combined_token_id
    mess3_vocab_size = T_mess3.shape[0]  # 3
    bloch_vocab_size = T_bloch.shape[0]  # 4
    bos_token_id = mess3_vocab_size * bloch_vocab_size  # 12 (for BOS)

    print(f"Tokenization: Mess3 vocab={mess3_vocab_size}, Bloch vocab={bloch_vocab_size}")
    print(f"Combined vocab size: {bos_token_id + (1 if bos else 0)}")

    for i, (mess3_seq, bloch_seq) in enumerate(zip(mess3_sequences, bloch_sequences)):
        combined_seq = []

        # Add BOS token if requested
        if bos:
            combined_seq.append(bos_token_id)

        # Convert each (mess3_token, bloch_token) pair to single token ID
        for m_tok, b_tok in zip(mess3_seq, bloch_seq):
            # Token ID: mess3_token + bloch_token * mess3_vocab_size
            combined_token = m_tok + b_tok * mess3_vocab_size
            combined_seq.append(combined_token)

        combined_seqs.append(combined_seq)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{num_sequences} sequence pairs")

    # Convert to tensors
    combined_seqs = torch.tensor(combined_seqs, dtype=torch.int32)

    # For independent sampling, assume uniform probabilities
    # (In practice, sequences have different probabilities based on the process dynamics)
    combined_probs = torch.ones(len(combined_seqs), dtype=torch.float32) / len(combined_seqs)

    print(f"\nFinal combined data:")
    print(f"Combined sequences shape: {combined_seqs.shape}")
    print(f"Combined probabilities shape: {combined_probs.shape}")
    print(f"Probability sum: {combined_probs.sum():.6f} (should be ~1.0)")

    # Calculate vocabulary size
    d_vocab = mess3_vocab_size * bloch_vocab_size  # 3 * 4 = 12
    if bos:
        d_vocab += 1  # Add BOS token

    print(f"Combined vocabulary size: {d_vocab}")
    print(f"Actual tokens in data: {torch.unique(combined_seqs)}")

    # Show example sequences
    print(f"\nExample sequences (first 3):")
    for i in range(min(3, len(combined_seqs))):
        seq = combined_seqs[i].tolist()
        print(f"Seq {i+1}: {seq}")
        if bos:
            mess3_part = [tok % mess3_vocab_size for tok in seq[1:]]  # Skip BOS
            bloch_part = [tok // mess3_vocab_size for tok in seq[1:]]  # Skip BOS
            print(f"        Mess3: [BOS] + {mess3_part}")
            print(f"        Bloch: [BOS] + {bloch_part}")

    return combined_seqs, combined_probs, d_vocab

def create_multipartite_dataloader(n_ctx=8, bos=True, num_sequences=10000,
                                 batches_per_epoch=200, batch_size=128, device='cuda', seed=42):
    """
    Create dataloader for multipartite training data.

    Args:
        n_ctx (int): Context length
        bos (bool): Include BOS token
        num_sequences (int): Number of independent sequence pairs to generate
        batches_per_epoch (int): Batches per epoch
        batch_size (int): Batch size
        device (str): Device
        seed (int): Random seed

    Returns:
        tuple: (dataloader, d_vocab)
    """
    combined_seqs, combined_probs, d_vocab = create_multipartite_data(
        n_ctx=n_ctx, bos=bos, num_sequences=num_sequences, device=device, seed=seed
    )

    print(f"\nCreating dataloader...")
    dataloader, d_vocab = get_dataloader_from_data(
        combined_seqs, combined_probs, batches_per_epoch, batch_size, device
    )

    print(f"Dataloader created with vocabulary size: {d_vocab}")

    return dataloader, d_vocab

def test_multipartite_data():
    """Test the multipartite data generation."""
    print("Testing Multipartite Data Generation")
    print("=" * 50)

    # Small test with fewer sequences
    combined_seqs, combined_probs, d_vocab = create_multipartite_data(
        n_ctx=3, bos=True, num_sequences=100, device='cpu', seed=42
    )

    print(f"\nVocabulary verification:")
    print(f"Expected vocab size: 13 (12 + BOS)")
    print(f"Actual vocab size: {d_vocab}")
    print(f"Unique tokens: {sorted(torch.unique(combined_seqs).tolist())}")

    return combined_seqs, combined_probs, d_vocab

if __name__ == "__main__":
    test_multipartite_data()