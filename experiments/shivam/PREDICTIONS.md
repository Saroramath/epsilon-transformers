# Pre-Registered Predictions: Multipartite Process Geometry

**Date:** 2025-10-19
**Honor Code:** These predictions are made BEFORE training and analyzing the model.

## Task Summary
Training a transformer on sequences from a multipartite process:
- **Mess3** (3 tokens, 3 states) ⊗ **Bloch Walk** (4 tokens, 4 states)
- Combined vocabulary: **12 tokens**
- Sequences generated **independently** for each process, then combined
- Small model: 2 layers, context window ~6-8

## Prediction 1: Independent Belief State Geometries
**Prediction:** The activation space will contain BOTH Mess3 and Bloch Walk belief state geometries as **independent subspaces**.

**Reasoning:**
- Since the sequences are generated independently, there's no interaction between processes
- The model should learn to track beliefs for each process separately
- Linear regression should recover both 3D (Mess3) and 4D (Bloch Walk) belief states

**Test:**
- Train separate regressors: activations → Mess3 beliefs AND activations → Bloch Walk beliefs
- Both should achieve high R² (>0.85)
- The regression weights should be approximately orthogonal (dot product ≈ 0)

## Prediction 2: Sierpiński Triangle for Mess3 Component
**Prediction:** When we project activations to the Mess3 belief space and visualize using **simplex projection**, we WILL see the characteristic Sierp

inski triangle fractal.

**Reasoning:**
- Mess3 has known fractal belief state geometry
- CRITICAL: Must use barycentric coordinates (`project_to_simplex_2d`) NOT raw dimensions
- If we use exhaustive sequence generation, the fractal should be preserved

**Test:**
- Extract Mess3 beliefs from activations via regression
- Project to 2D using simplex projection
- Visual inspection for fractal structure
- Compare to ground-truth Mess3 beliefs (should show fractals)

## Prediction 3: Dimensionality of Activation Space
**Prediction:** The effective dimensionality of activations should be **~7 dimensions** (3 for Mess3 + 4 for Bloch Walk).

**Reasoning:**
- Each process contributes its hidden state dimensionality
- No interaction between processes → additive dimensionality
- PCA should show first 7 components explain >90% variance

**Test:**
- PCA on activations
- Check cumulative variance explained by first 7 PCs
- Scree plot should show elbow around 7

## Prediction 4: Orthogonality of Process Representations
**Prediction:** The subspaces encoding Mess3 vs Bloch Walk beliefs will be **approximately orthogonal**.

**Reasoning:**
- No causal relationship between processes
- Model has no incentive to entangle them
- Orthogonal representations are more efficient

**Test:**
- Compute regression weights: W_mess3 (64×3) and W_bloch (64×4)
- Compute orthogonality: ||W_mess3^T @ W_bloch||_F should be small
- Cosine similarity between subspaces should be near 0

## Prediction 5: What We Might Be Wrong About
**Alternative hypotheses to consider:**

1. **Entangled Representations:** The model might learn entangled representations if there are spurious correlations in the training data

2. **No Fractals:** The fractal structure might be lost if:
   - Training is insufficient (need to reach optimal loss)
   - Model capacity is too small
   - We use wrong visualization (must use simplex projection!)

3. **Higher Dimensionality:** Might need >7 dimensions if model learns redundant features

4. **Non-linear Mixing:** Beliefs might not be linearly decodable if model uses non-linear encoding

## Key Experimental Controls
- Use **exhaustive sequence generation** (not sampling!)
- Train until **normalized loss ≈ 1.0** (optimal)
- Use **simplex projection** for visualization
- Check **both** PCA and direct regression approaches

---

**Signature:** Pre-registered predictions made before running experiments
