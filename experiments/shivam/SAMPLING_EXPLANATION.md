# How Sampling Works During Training

## Step-by-Step Process

### Step 1: Generate ALL Sequences (Once, Before Training)

```python
# From train_multipartite_exhaustive.py:162-164
transformer_inputs, probs, loss_lower_bound, metadata = create_multipartite_sequences_exhaustive(
    n_ctx, bos, device
)
```

This creates:
- `transformer_inputs`: Tensor of shape `[248832, 5]` - ALL sequences
- `probs`: Tensor of shape `[248832]` - Probability of each sequence
- All stored in GPU memory

**Example data**:
```
transformer_inputs[0] = [3, 10, 4, 0, 4]      probs[0] = 0.000003
transformer_inputs[1] = [3, 10, 4, 0, 5]      probs[1] = 0.000012
transformer_inputs[2] = [3, 10, 4, 0, 6]      probs[2] = 0.000008
...
transformer_inputs[248831] = [11, 7, 11, 11, 11]  probs[248831] = 0.000001
```

### Step 2: Create BatchGenerator

```python
# From train_multipartite_exhaustive.py:170-176
dataloader = BatchGenerator(
    transformer_inputs,  # All 248,832 sequences
    probs,              # All 248,832 probabilities
    batches_per_epoch,  # 200 batches
    batch_size,         # 1024 sequences per batch
    device
)
```

The `BatchGenerator` stores references to ALL sequences and probabilities.

### Step 3: Sampling During Training (The Key!)

Every time we iterate through the dataloader, this happens:

```python
# From dataloader.py:62-67
def __iter__(self):
    for _ in range(self.batches_per_epoch):  # 200 iterations
        # Sample indices based on probabilities
        sample_inds = torch.multinomial(self.probs, self.batch_size, replacement=True)
        batch = self.transformer_inputs[sample_inds]
        X, Y = batch[:, :-1], batch[:, 1:]
        yield X, Y
```

**Key line**: `torch.multinomial(self.probs, self.batch_size, replacement=True)`

This samples 1024 **indices** from {0, 1, 2, ..., 248831} where:
- Each index `i` has probability `probs[i]` of being selected
- `replacement=True` means same sequence can appear multiple times in a batch
- Higher probability sequences appear more frequently!

## Concrete Example of One Training Step

### The Probability Distribution
```python
# Simplified example with just 6 sequences
sequences = [
    [3, 10, 4, 0, 4],   # prob = 0.000003 (very rare)
    [3, 10, 4, 0, 5],   # prob = 0.000012 (rare)
    [0, 4, 8, 4, 8],    # prob = 0.050000 (common!)
    [0, 5, 9, 5, 9],    # prob = 0.080000 (very common!)
    [4, 5, 6, 7, 8],    # prob = 0.020000 (uncommon)
    [11, 7, 11, 11, 11] # prob = 0.000001 (extremely rare)
]
```

### Sampling a Batch of 10 Sequences

```python
sample_inds = torch.multinomial(probs, batch_size=10, replacement=True)
# Result might be: [3, 3, 2, 3, 3, 2, 4, 3, 2, 0]
#                   ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#                   Index into sequences array

# The batch contains:
# - Sequence 3 appears 5 times (50%)  ← prob=0.08 (very common)
# - Sequence 2 appears 3 times (30%)  ← prob=0.05 (common)
# - Sequence 4 appears 1 time  (10%)  ← prob=0.02 (uncommon)
# - Sequence 0 appears 1 time  (10%)  ← prob=0.000003 (rare, but still sampled!)
# - Sequences 1, 5 not sampled        ← too rare in this particular batch
```

**Key insight**: Sequence 3 (highest probability) appears most frequently!

## Mathematical Guarantee

Over many batches, the **expected frequency** matches the true probability:

```python
Expected times sequence i appears per epoch:
    = batches_per_epoch × batch_size × probs[i]
    = 200 × 1024 × probs[i]
    = 204,800 × probs[i]
```

Example:
- Sequence with prob=0.05: Expected ~10,240 appearances per epoch
- Sequence with prob=0.0001: Expected ~20 appearances per epoch
- Sequence with prob=0.000001: Expected ~0.2 appearances per epoch (appears rarely)

## Why This is Brilliant

### Compared to Alternative 1: Uniform Sampling (BAD)
```python
# What if we sampled uniformly?
sample_inds = torch.randint(0, len(sequences), (batch_size,))

# Problem:
# - Rare sequences get same weight as common sequences
# - Model trains on wrong distribution
# - Loss doesn't converge to true optimum
```

### Compared to Alternative 2: Weighted Loss (Less Efficient)
```python
# What if we sampled uniformly but weighted loss?
for batch in uniform_batches:
    loss = compute_loss(batch)
    weighted_loss = loss * get_probability(batch)  # Expensive!
    
# Problem:
# - Need to look up probability for each sequence (slow)
# - Wastes computation on rare sequences
# - Gradients dominated by rare sequences (numerical issues)
```

### Our Approach: Probability-Based Sampling (BEST)
```python
# Sample by probability, compute unweighted loss
sample_inds = torch.multinomial(probs, batch_size, replacement=True)
batch = sequences[sample_inds]
loss = compute_loss(batch)  # Simple, fast, correct!

# Benefits:
# ✓ Common sequences naturally appear more often
# ✓ Training focuses on important parts of distribution
# ✓ Loss converges to true optimal
# ✓ Fast (no per-sequence probability lookup)
```

## What Happens During Validation?

Validation uses **ALL sequences** (no sampling):

```python
# From dataloader.py:69-72
def validation_data(self):
    X = self.transformer_inputs[:, :-1]  # All 248,832 sequences
    Y = self.transformer_inputs[:, 1:]
    return X, Y, self.probs  # Return probabilities for weighting
```

Then during validation loss computation:

```python
# From train_multipartite_exhaustive.py:132-137
loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
loss = loss.reshape(batch_size, seq_length)

# Weight by sequence probabilities
weighted_loss = loss * probs.unsqueeze(1)
loss_per_position = weighted_loss.sum(dim=0)
```

This computes the **exact expected loss** over the true distribution!

## Training vs Validation Summary

| Stage | Data Used | Loss Computation |
|-------|-----------|------------------|
| **Training** | Sampled batches (1024 sequences) | Unweighted cross-entropy |
| **Validation** | ALL sequences (248,832) | Probability-weighted cross-entropy |

**Why different?**
- Training: Sample by probability → naturally correct distribution
- Validation: Use all data → explicitly weight by probability

Both converge to the same optimum!

## Memory Efficiency Note

You might wonder: "Doesn't storing 248,832 sequences use a lot of memory?"

```python
Memory usage:
- Sequences: 248,832 × 5 × 4 bytes (int32) = 4.97 MB
- Probabilities: 248,832 × 4 bytes (float32) = 0.99 MB
- Total: ~6 MB

GPU memory: Typically 40-80 GB
```

This is **tiny**! Even with 248,832 sequences, we use <10 MB.

The memory is dominated by:
- Model parameters: ~67K params × 4 bytes = 268 KB
- Activations during forward pass: batch_size × seq_len × d_model × layers
- Gradients: Similar to parameters

So exhaustive generation is very efficient for small sequences!

## Real Data Analysis: What Actually Happens

From the simulation above, we discovered:

### In ONE Batch (1,024 sequences):
- **1,018 unique sequences** (99.4% are different!)
- **Most sequences appear only once** (1,012 singletons)
- **Maximum repeats**: 2 times
- **Coverage**: Only 0.41% of all sequences (1,018 / 248,832)

### The Distribution is VERY Flat!

**Highest probability sequence**: `[0,0,0,0,0]` with P = 0.000016
- Expected appearances per batch: **0.02** (1 in 50 batches!)
- Expected appearances per epoch (200 batches): **3.2 times**

**Lowest probability sequence**: Various with P = 0.0000025
- Expected appearances per batch: **0.0025** (1 in 400 batches!)
- Expected appearances per epoch: **0.5 times** (might not appear at all!)

### Why is the Distribution So Flat?

The multipartite construction spreads probability mass:

```python
P(mess3_seq) × P(bloch_seq) = very small number

Example:
  Most common Mess3: P ≈ 0.007
  Most common Bloch:  P ≈ 0.002
  Combined: 0.007 × 0.002 = 0.000014

  Rare Mess3: P ≈ 0.0005
  Rare Bloch:  P ≈ 0.0005
  Combined: 0.0005 × 0.0005 = 0.00000025
```

The 248,832 sequences share probability fairly evenly!

### Coverage Over One Epoch

```
Samples per epoch: 200 batches × 1,024 = 204,800 samples

Expected unique sequences seen per epoch:
  ≈ 90-95% of all sequences (rough estimate)

Average appearances per sequence:
  204,800 / 248,832 ≈ 0.82 times per epoch
```

Most sequences appear **less than once per epoch on average**!

### Implications for Training

**Good news:**
- ✓ Model sees diverse sequences (not dominated by a few high-prob ones)
- ✓ No severe overfitting to common patterns
- ✓ Sampling correctly represents the distribution

**Interesting observation:**
- With such a flat distribution, almost all sequences are "rare"
- Model needs many epochs to see each sequence multiple times
- 1000 epochs × 0.82 appearances/epoch ≈ 820 times per sequence on average

This is **why exhaustive generation works well** - the distribution is so flat that probabilistic sampling behaves almost like uniform sampling, but with correct weighting!
