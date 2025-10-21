# Control Experiment Results: Individual vs Multipartite Training

**Date:** 2025-10-20  
**Question:** Does multipartite combination prevent learning belief states?  
**Answer:** **NO** - Individual training shows identical poor results!

## Experimental Setup

### Individual Training (Control)
- **Mess3 only**: 243 sequences, vocab=3, n_ctx=4
- **Bloch Walk only**: 1,024 sequences, vocab=4, n_ctx=4
- Same architecture as multipartite: 2 layers, d_model=64

### Multipartite Training (Original)
- **Mess3 ⊗ Bloch**: 248,832 sequences, vocab=12, n_ctx=4
- Combined process with independent generation

## Results Comparison

### Next-Token Prediction (All Optimal!)

| Setup | Normalized Loss | Interpretation |
|-------|----------------|----------------|
| Mess3 individual | 1.0000 | 100% optimal |
| Bloch individual | 1.0000 | 100% optimal |
| Multipartite | 1.0001 | 99.99% optimal |

**Finding:** All models achieve perfect next-token prediction!

### Belief State Recovery (All Poor!)

| Setup | Mess3 R² | Bloch R² | Dimensionality |
|-------|----------|----------|----------------|
| **Individual** | 0.206 | 0.464 | Mess3: 5, Bloch: 2 |
| **Multipartite** | 0.204 | 0.462 | - |
| **Difference** | +0.002 | +0.002 | Negligible |

**Finding:** Individual training provides **NO improvement** in belief state learning!

### Geometric Structure

**Mess3 (Expected: Sierpiński Triangle Fractal)**
- ✅ Ground truth: Perfect fractal structure
- ❌ Individual predictions: Scattered noise, no fractal (R²=0.206)
- ❌ Multipartite predictions: Scattered noise, no fractal (R²=0.204)

**Bloch Walk (Expected: Circular/Elliptical)**
- ✅ Ground truth: Clear discrete clusters in circular pattern
- ⚠️ Individual predictions: Rough clusters, partial structure (R²=0.464)
- ⚠️ Multipartite predictions: Rough clusters, partial structure (R²=0.462)

**Finding:** Fractal structure completely lost in both cases!

## Key Insights

### 1. Multipartite Is NOT The Problem
The nearly identical R² scores (±0.002) prove that combining processes does NOT cause the poor belief state representations.

### 2. Context Length Is Likely The Problem
With n_ctx=4:
- **Mess3**: Only 243 unique sequences (3^5)
- **Bloch**: Only 1,024 unique sequences (4^5)

The transformer can **memorize** these patterns without learning the underlying belief state dynamics!

**Evidence:**
- Perfect next-token prediction (optimal loss)
- Poor belief state recovery (low R²)
- Missing fractal structure
- Model doesn't need Bayesian inference to predict tokens

### 3. Memorization vs Generalization
Current results suggest the model is using a **lookup table strategy**:
```python
# What model might be doing:
pattern_table = {
    "0120": predict(1, prob=0.7),  # Direct memorization
    "1201": predict(0, prob=0.4),
    # ... 243 entries (fits easily in 64-dim space)
}

# vs what we WANT:
belief = bayesian_update(sequence)  # 3D simplex dynamics
prediction = T @ belief
```

### 4. Dimensionality Mismatch
- **Mess3**: Uses 5 dimensions (expected 3!) → Overcomplete representation
- **Bloch**: Uses 2 dimensions (expected 3!) → Undercomplete representation

Neither matches the theoretical 3-state hidden process.

## Recommendations

### Immediate Test: Increase Context Length
```python
n_ctx = 8  # 3^9 = 19,683 sequences (Mess3)
           # 4^9 = 262,144 sequences (Bloch)
```

**Prediction:**
- If context=4 enables memorization → longer context should FORCE belief state learning
- R² should improve significantly
- Fractal structure should emerge

### Generalization Test
```python
# Train on n_ctx=4
# Test prediction on n_ctx=12 (longer, unseen sequences)
```

**Prediction:**
- Memorization strategy: Loss degrades on longer sequences
- Belief state strategy: Generalizes to any length

### Process Interpolation Test
```python
# Train on Mess3(x=0.15, a=0.6)
# Test on Mess3(x=0.20, a=0.6) 
```

**Prediction:**
- Memorization: Complete failure
- Belief states: Partial generalization

## Conclusion

**The hypothesis that multipartite combination prevents belief state learning is REJECTED.**

Individual training produces **identical results** to multipartite training:
- ✅ Same optimal next-token prediction
- ❌ Same poor belief state R²
- ❌ Same loss of fractal geometry

**Root cause hypothesis:** Short context length (n_ctx=4) allows memorization of 243-1024 patterns without learning the underlying belief state dynamics.

**Next steps:** Rerun experiments with n_ctx=8 or n_ctx=12 to test if longer contexts force belief state learning.

---

## Comparison with Official Configs

The official `experiment_config_transformer_mess3_bloch.yaml` uses **n_ctx=8**, while these experiments used **n_ctx=4**.

This difference likely explains why belief states aren't learned - the sequence space is too small.
