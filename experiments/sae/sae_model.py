"""
Sparse Autoencoder (SAE) Model

Architecture:
- Encoder: Linear layer with ReLU activation
- Sparsity: L1 penalty on hidden activations
- Decoder: Linear layer (tied or untied weights)
- Bias: Pre-encoder bias to center activations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features in activations.

    Following standard SAE architecture:
    - Pre-encoder bias for centering
    - Encoder with ReLU activation
    - L1 sparsity penalty
    - Decoder (optionally tied weights)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        tie_weights: bool = False,
        device: str = 'cuda'
    ):
        """
        Args:
            d_model: Input/output dimension (transformer d_model)
            d_hidden: Hidden dimension (typically 4-8x d_model for sparsity)
            tie_weights: If True, decoder = encoder.T (reduces parameters)
            device: Device to place model on
        """
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.tie_weights = tie_weights
        self.device = device

        # Pre-encoder bias (centers activations)
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        # Encoder
        self.W_enc = nn.Parameter(torch.randn(d_model, d_hidden) / (d_model ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))

        # Decoder
        if tie_weights:
            # Tied weights: W_dec = W_enc.T
            self.W_dec = None
        else:
            self.W_dec = nn.Parameter(torch.randn(d_hidden, d_model) / (d_hidden ** 0.5))

        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self.to(device)

    def encode(self, x):
        """
        Encode input to sparse hidden representation.

        Args:
            x: Input activations [batch, d_model]

        Returns:
            h: Sparse hidden activations [batch, d_hidden]
        """
        # Center input
        x_centered = x - self.b_pre

        # Linear + ReLU
        h = F.relu(x_centered @ self.W_enc + self.b_enc)

        return h

    def decode(self, h):
        """
        Decode sparse representation back to input space.

        Args:
            h: Sparse hidden activations [batch, d_hidden]

        Returns:
            x_recon: Reconstructed activations [batch, d_model]
        """
        if self.tie_weights:
            # Use tied weights: W_dec = W_enc.T
            x_recon = h @ self.W_enc.T + self.b_dec
        else:
            x_recon = h @ self.W_dec + self.b_dec

        return x_recon

    def forward(self, x):
        """
        Full forward pass: encode + decode.

        Args:
            x: Input activations [batch, d_model]

        Returns:
            x_recon: Reconstructed activations [batch, d_model]
            h: Sparse hidden activations [batch, d_hidden]
        """
        h = self.encode(x)
        x_recon = self.decode(h)

        return x_recon, h

    def loss(self, x, sparsity_coeff: float = 0.01):
        """
        Compute SAE loss: reconstruction + sparsity.

        Loss = MSE(x, x_recon) + β * L1(h)

        Args:
            x: Input activations [batch, d_model]
            sparsity_coeff: Coefficient for L1 sparsity penalty (β)

        Returns:
            total_loss: Combined loss
            recon_loss: MSE reconstruction loss
            sparsity_loss: L1 sparsity loss
        """
        x_recon, h = self.forward(x)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # Sparsity loss (L1 on hidden activations)
        sparsity_loss = h.abs().mean()

        # Total loss
        total_loss = recon_loss + sparsity_coeff * sparsity_loss

        return total_loss, recon_loss, sparsity_loss

    def get_feature_stats(self, x):
        """
        Compute statistics about feature activations.

        Useful for analyzing which features are active and how sparse they are.

        Args:
            x: Input activations [batch, d_model]

        Returns:
            dict with:
                - activation_freq: Fraction of examples where each feature is active
                - mean_activation: Mean activation value per feature (when active)
                - max_activation: Max activation value per feature
        """
        _, h = self.forward(x)

        # Which features are active (h > 0)?
        is_active = (h > 0).float()

        activation_freq = is_active.mean(dim=0)  # [d_hidden]
        mean_activation = h.mean(dim=0)  # [d_hidden]
        max_activation = h.max(dim=0)[0]  # [d_hidden]

        return {
            'activation_freq': activation_freq.detach().cpu().numpy(),
            'mean_activation': mean_activation.detach().cpu().numpy(),
            'max_activation': max_activation.detach().cpu().numpy(),
            'sparsity': (h > 0).float().mean().item()  # Overall sparsity
        }

    def get_decoder_weights(self):
        """
        Get decoder weight matrix.

        Returns:
            W_dec: Decoder weights [d_hidden, d_model]
        """
        if self.tie_weights:
            return self.W_enc.T.detach().cpu().numpy()
        else:
            return self.W_dec.detach().cpu().numpy()

    def get_encoder_weights(self):
        """
        Get encoder weight matrix.

        Returns:
            W_enc: Encoder weights [d_model, d_hidden]
        """
        return self.W_enc.detach().cpu().numpy()


def initialize_sae_from_pca(activations, d_hidden, device='cuda'):
    """
    Initialize SAE weights using PCA.

    This can provide a better starting point than random initialization.

    Args:
        activations: Training activations [n_samples, d_model]
        d_hidden: SAE hidden dimension
        device: Device to place SAE on

    Returns:
        sae: Initialized SparseAutoencoder
    """
    d_model = activations.shape[1]

    # Center activations
    mean = activations.mean(dim=0)
    activations_centered = activations - mean

    # Compute PCA
    U, S, Vt = torch.pca_lowrank(activations_centered, q=min(d_hidden, d_model))

    # Create SAE
    sae = SparseAutoencoder(d_model, d_hidden, device=device)

    # Initialize encoder with PCA components
    n_components = min(d_hidden, Vt.shape[0])
    sae.W_enc.data[:, :n_components] = Vt[:n_components].T.to(device)

    # Initialize pre-encoder bias with mean
    sae.b_pre.data = mean.to(device)

    # Initialize decoder (transpose of encoder if tied)
    if not sae.tie_weights:
        sae.W_dec.data[:n_components, :] = Vt[:n_components].to(device)

    return sae
