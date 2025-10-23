#!/usr/bin/env python3
"""
Setup script for the Multipartite Mess3 × Bloch Walk Experiment

This script configures Weights & Biases, sets up the multipartite process,
and prepares the research environment for training transformers on
Cartesian product sequences from Mess3 and Bloch Walk processes.

Author: Claude Code
Date: 2025-10-20
"""

import os
import sys
import wandb
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

def setup_wandb():
    """Configure Weights & Biases with API key."""
    wandb_key_file = Path(__file__).parent / ".wandb_key"

    if wandb_key_file.exists():
        with open(wandb_key_file, 'r') as f:
            api_key = f.read().strip()

        # Set environment variable
        os.environ['WANDB_API_KEY'] = api_key

        # Login to wandb
        wandb.login(key=api_key)
        print("✅ Weights & Biases configured successfully!")
        return True
    else:
        print("❌ WANDB API key file not found. Please save your key to .wandb_key")
        return False

def setup_huggingface():
    """Configure Hugging Face Hub with API key."""
    hf_key_file = Path(__file__).parent / ".hf_key"

    if hf_key_file.exists():
        with open(hf_key_file, 'r') as f:
            api_key = f.read().strip()

        # Set environment variable
        os.environ['HUGGINGFACE_HUB_TOKEN'] = api_key
        print("✅ Hugging Face Hub configured successfully!")
        return True
    else:
        print("❌ HuggingFace API key file not found. Please save your key to .hf_key")
        return False

def check_gpu():
    """Check GPU availability and configuration."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"✅ GPU memory: {gpu_memory:.1f} GB")
    else:
        device = "cpu"
        print("⚠️ No GPU detected. Training will use CPU.")

    return device

def create_project_structure():
    """Create necessary directories for the experiment."""
    base_dir = Path(__file__).parent

    directories = [
        "checkpoints",
        "figures",
        "logs",
        "data",
        "configs",
        "results"
    ]

    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def main():
    """Main setup function."""
    print("🚀 Setting up Multipartite Mess3 × Bloch Walk Experiment\n")

    # Setup APIs
    wandb_ok = setup_wandb()
    hf_ok = setup_huggingface()

    # Check hardware
    device = check_gpu()

    # Create project structure
    create_project_structure()

    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    print(f"Weights & Biases: {'✅ Ready' if wandb_ok else '❌ Failed'}")
    print(f"Hugging Face:     {'✅ Ready' if hf_ok else '❌ Failed'}")
    print(f"Device:           {device}")
    print(f"Working Directory: {Path(__file__).parent}")

    if wandb_ok and hf_ok:
        print("\n🎉 All systems ready for multipartite training!")
        return True
    else:
        print("\n❌ Some components failed to configure.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)