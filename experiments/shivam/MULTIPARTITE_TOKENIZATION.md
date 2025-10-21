# Multipartite Dataset Generation & Tokenization

## High-Level Process

```python
# Step 1: Generate ALL sequences for each process
mess3_seqs = generate_all_seqs(Mess3, n_ctx=4)   # 243 sequences (3^5)
bloch_seqs = generate_all_seqs(Bloch, n_ctx=4)   # 1,024 sequences (4^5)

# Step 2: Create Cartesian product
for each mess3_seq in mess3_seqs:
    for each bloch_seq in bloch_seqs:
        combine(mess3_seq, bloch_seq)  # 243 × 1,024 = 248,832 pairs

# Step 3: Compute joint probabilities (independent processes)
P(mess3, bloch) = P(mess3) * P(bloch)
```

## Tokenization Scheme

**Key line** (`train_multipartite_exhaustive.py:79`):
```python
multipartite_token = mess3_token * 4 + bloch_token
```

This is a **paired encoding** that maps 2D pairs to 1D tokens:

### Token Space
- **Mess3 tokens**: {0, 1, 2} (3 tokens)
- **Bloch tokens**: {0, 1, 2, 3} (4 tokens)
- **Multipartite tokens**: {0, 1, 2, ..., 11} (12 tokens)

### Encoding Table

| Mess3 | Bloch | Formula | Combined Token |
|-------|-------|---------|----------------|
| 0 | 0 | 0×4 + 0 | **0** |
| 0 | 1 | 0×4 + 1 | **1** |
| 0 | 2 | 0×4 + 2 | **2** |
| 0 | 3 | 0×4 + 3 | **3** |
| 1 | 0 | 1×4 + 0 | **4** |
| 1 | 1 | 1×4 + 1 | **5** |
| 1 | 2 | 1×4 + 2 | **6** |
| 1 | 3 | 1×4 + 3 | **7** |
| 2 | 0 | 2×4 + 0 | **8** |
| 2 | 1 | 2×4 + 1 | **9** |
| 2 | 2 | 2×4 + 2 | **10** |
| 2 | 3 | 2×4 + 3 | **11** |

### Decoding Formula
```python
# To recover original tokens from combined token:
mess3_token = multipartite_token // 4  # Integer division
bloch_token = multipartite_token % 4   # Modulo
```

### Concrete Example

**Mess3 sequence**: [0, 1, 2, 0, 1]  
**Bloch sequence**: [3, 2, 1, 0, 3]

**Combined multipartite sequence**:
```python
position 0: mess3=0, bloch=3 → 0*4 + 3 = 3
position 1: mess3=1, bloch=2 → 1*4 + 2 = 6
position 2: mess3=2, bloch=1 → 2*4 + 1 = 9
position 3: mess3=0, bloch=0 → 0*4 + 0 = 0
position 4: mess3=1, bloch=3 → 1*4 + 3 = 7

Multipartite sequence: [3, 6, 9, 0, 7]
```

## Probability Calculation

Since processes are **independent**:

```python
P(mess3_seq, bloch_seq) = P(mess3_seq) * P(bloch_seq)
```

**Example**:
```python
mess3_seq = [0,1,2,0,1] → P(mess3) = 0.042
bloch_seq = [3,2,1,0,3] → P(bloch) = 0.015

Combined sequence: [3,6,9,0,7] → P(combined) = 0.042 * 0.015 = 0.00063
```

## Dataset Statistics

### Before Combination
- Mess3: 243 sequences, each 5 tokens long
- Bloch: 1,024 sequences, each 5 tokens long

### After Combination
- **Total sequences**: 243 × 1,024 = **248,832**
- **Sequence length**: 5 tokens
- **Vocabulary size**: 12 tokens (3 × 4)
- **All probabilities sum to**: 1.0 (verified in code)

### Loss Lower Bound
Since processes are independent, entropy is additive:
```python
H(Mess3, Bloch) = H(Mess3) + H(Bloch)
loss_lower_bound = mess3_lb + bloch_lb
```

## Why This Encoding?

### Advantages
1. **Compact**: 12 tokens instead of separate embeddings
2. **Deterministic**: Unique encoding/decoding
3. **Preserves independence**: Can extract Mess3/Bloch parts via `// 4` and `% 4`

### Disadvantages (Relevant to Your Findings!)
1. **Imbalanced magnitude**: Mess3 gets 4× weight (multiplied by 4)
   - Token 0-3: Mess3=0
   - Token 4-7: Mess3=1
   - Token 8-11: Mess3=2
   
2. **Non-symmetric**: Mess3 vs Bloch treated differently in encoding

3. **May bias learning**: Model might learn Mess3 better due to larger magnitude

   **Evidence from your results**:
   - Mess3 R² = 0.204 (poor)
   - Bloch R² = 0.462 (better!)
   - Opposite of what you'd expect from 4× weighting!

## Alternative Encodings (Not Used)

### Option 1: Separate Embeddings
```python
# Two separate token sequences
mess3_embedding = Embed(mess3_tokens)  # [batch, seq, d_model/2]
bloch_embedding = Embed(bloch_tokens)  # [batch, seq, d_model/2]
combined = concat([mess3_embedding, bloch_embedding])
```

### Option 2: Symmetric Encoding
```python
# Use larger vocabulary to avoid asymmetry
multipartite_token = mess3_token * 10 + bloch_token  # 30 tokens instead of 12
```

### Option 3: Learned Pairing
```python
# Let model learn how to combine
combined_embedding = f(Embed(mess3), Embed(bloch))
```

## Connection to Your Hypothesis

The tokenization `mess3_token * 4 + bloch_token` creates an **asymmetry**:

- Mess3 changes affect higher-order bits (multiples of 4)
- Bloch changes affect lower-order bits (0-3 within each group)

This could explain why:
1. Both processes are learned poorly (similar R²)
2. The encoding makes it harder to learn independent belief states
3. Model might focus on memorizing the 12-token patterns directly

**Recommendation**: Try symmetric encoding or separate embeddings to test if this affects belief state learning!
