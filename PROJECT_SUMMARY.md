# CS441 Trainable Sparse Attention - Project Summary

## 🎯 Project Overview

This project implements **trainable sparse attention adapters** for Llama 3.2 1B, inspired by Native Sparse Attention and SoftCoT pipelines.

### Key Innovation

**Problem**: Native Sparse Attention trains models from scratch; SoftCoT uses auxiliary models. Neither is parameter-efficient for existing LLMs.

**Solution**: Add lightweight trainable sparse attention adapters to frozen Llama 3.2 1B, enabling sparse attention with only 1-12% additional parameters.

## 📊 Architecture Comparison

| Aspect | Native Sparse Attn | SoftCoT | **This Project (CS441)** |
|--------|-------------------|---------|-------------------------|
| Base Model | Train from scratch | Frozen (8B) | **Frozen (1B)** |
| Additional Modules | Integrated in layers | Assistant model (1B) | **Sparse adapters** |
| Training Target | All parameters | Projection (~0.5%) | **Adapters (~1-12%)** |
| Sparse Mechanism | 3-branch attention | Soft tokens | **3-branch adapters** |
| Inference Speed | Faster (with kernels) | Same/Slower | **Faster (theoretically)** |

## 🏗️ Model Architecture

```
Input Tokens
    ↓
┌─────────────────────────────────────────┐
│  Frozen Llama 3.2 1B Backbone           │
│                                         │
│  For each layer (0-15):                 │
│    ├─ Self-Attention (frozen)          │
│    │  └─> Produces Q, K, V             │
│    │                                    │
│    ├─ Sparse Attention Adapter 🔥      │
│    │  ├─ Branch 1: Compressed Attention│
│    │  │  ├─ Compress K (max pool)      │
│    │  │  ├─ Compress V (MLP)           │
│    │  │  └─ Coarse attention           │
│    │  │                                 │
│    │  ├─ Branch 2: Fine Selection      │
│    │  │  ├─ Select top-k blocks        │
│    │  │  └─ Fine attention             │
│    │  │                                 │
│    │  ├─ Branch 3: Sliding Window      │
│    │  │  └─ Local attention            │
│    │  │                                 │
│    │  └─ Strategy Combiner (learned)   │
│    │     └─> Weighted combination      │
│    │                                    │
│    └─ Feed-Forward (frozen)            │
└─────────────────────────────────────────┘
    ↓
Output Logits
```

## 📁 Project Structure

```
train/
├── sparse_attention_model.py       # Core model implementation
│   ├── MaxPoolCompress             # K compression (max pooling)
│   ├── MLPCompress                 # V compression (MLP)
│   ├── SparseAttentionAdapter      # Per-layer adapter
│   └── LlamaWithSparseAttention    # Full model wrapper
│
├── train_sparse_attention.py       # Training script
├── evaluate_sparse_attention.py    # Evaluation script
├── test_sparse_model.py           # Unit tests
│
├── run_sparse_train.sh            # Training launcher
├── run_sparse_eval.sh             # Evaluation launcher
│
├── README_SPARSE_ATTENTION.md      # User guide
├── DESIGN_GUIDE_CN.md             # Design details (Chinese)
└── PROJECT_SUMMARY.md             # This file
```

## 🔧 Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Create environment
conda create -n sparse-attn python=3.10 -y
conda activate sparse-attn

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==2.7.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install transformers==4.51.0 fastNLP==0.7.0
pip install -r requirements.txt
```

### Option 2: Using uv (Fast Alternative)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment with uv
uv venv sparse-attn --python 3.10
source sparse-attn/bin/activate  # On Linux/Mac
# sparse-attn\Scripts\activate  # On Windows

# Install dependencies with uv (much faster!)
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.51.0 fastNLP==0.7.0
uv pip install -r requirements.txt
```

### Option 3: Quick Install (If already have Python env)

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; import transformers; import fastNLP; print('✅ All dependencies installed!')"
```

## 🚀 Quick Start

### 1. Test Installation

```bash
python test_sparse_model.py
```

This will test:
- ✅ Compression modules (MaxPool & MLP)
- ✅ Sparse attention adapter
- ✅ Full model creation (optional)

### 2. Train on GSM8K

```bash
# Edit data path in the script
bash run_sparse_train.sh experiment1 gsm8k
```

Or with custom parameters:

```bash
python train_sparse_attention.py \
    --model_id meta-llama/Llama-3.2-1B \
    --output_name exp1 \
    --task_name gsm8k \
    --data_path /path/to/gsm8k \
    --batch_size 4 \
    --n_epochs 3 \
    --compress_block_size 16 \
    --num_selected_blocks 4 \
    --k_compress_method max_pool \
    --v_compress_method mlp
```

### 3. Evaluate

```bash
bash run_sparse_eval.sh ./ckpt/exp1-gsm8k-3epoch-Llama-3.2-1B-sparse gsm8k
```

## 🔬 Design Decisions

### Q1: Why Max Pooling for K?

**Rationale**:
- K is used for computing similarity scores
- Max pooling preserves most salient features
- Zero additional parameters (only positional embeddings)
- Empirically comparable to MLP in Native Sparse Attention

### Q2: Why MLP for V?

**Rationale**:
- V values are directly propagated to output
- Needs better representation capacity
- Cannot just take "max" - needs weighted aggregation
- Small MLP (block_size × dim_head → dim_head) is sufficient

### Q3: Parameter Count Analysis

For Llama 3.2 1B (hidden_size=2048, num_heads=32, num_kv_heads=8):

```python
Per-layer parameters:
├─ K compression (max pool): ~8K (pos embeddings only)
├─ V compression (MLP): ~9M (per-head MLPs)
├─ Strategy combiner: ~200K
└─ Total per layer: ~9.2M

16 layers total: ~147M parameters
Base model: ~1.2B parameters
Percentage: 147M / 1200M = 12.2%
```

**Optimization options**:
```python
# Option 1: Reduce expand_factor (recommended)
expand_factor = 0.5  # ~6% total parameters

# Option 2: Shared compression across heads
# ~1.5% total parameters

# Option 3: Smaller block size
block_size = 8  # ~3% total parameters
```

### Q4: Why Adapter Pattern?

**Alternatives considered**:

1. **Modify Transformers library directly** ❌
   - Hard to maintain
   - Breaks compatibility

2. **LoRA on projection matrices** ❌
   - Modifies weight matrices, not computation pattern
   - Cannot implement sparse attention logic

3. **Adapter modules** ✅
   - Plug-and-play
   - Maintains compatibility
   - Flexible architecture

## 🎓 Key Differences from References

### vs Native Sparse Attention

| Aspect | Native Sparse Attention | This Project |
|--------|------------------------|--------------|
| Training | From scratch | Frozen backbone |
| Integration | Built into transformer | Adapter modules |
| Parameters | ~100% trainable | ~1-12% trainable |
| Use Case | New models | Existing LLMs |

### vs SoftCoT

| Aspect | SoftCoT | This Project |
|--------|---------|--------------|
| Auxiliary Model | Yes (assistant) | No |
| Special Tokens | Yes (thought tokens) | No |
| Data Format | Custom | Standard LM |
| Speedup | None | Theoretical |
| Complexity | Two-model alignment | Single-model training |

## 📊 Expected Results

### Performance Targets

- **Accuracy**: ~95% of full attention baseline
- **Speed**: 2-3x faster (with CUDA kernels)
- **Memory**: Similar or better
- **Parameters**: Only ~1-12% additional

### Ablation Studies to Run

1. **Compression methods**:
   ```bash
   k=max_pool, v=mlp  (recommended)
   k=mlp, v=mlp       (more parameters)
   k=max_pool, v=max_pool  (fewer parameters)
   ```

2. **Block sizes**:
   ```bash
   compress_block_size ∈ {8, 16, 32}
   num_selected_blocks ∈ {2, 4, 8}
   ```

3. **Strategy weights**:
   - Track learned weights over training
   - Analyze which branch is most used

## ⚠️ Known Limitations

### 1. Forward Pass Integration (CRITICAL)

**Status**: ⬜ Not yet implemented

**Issue**: Adapter modules are created but not integrated into Llama's forward pass.

**Solution needed**: Implement hooks or monkey-patching to:
```python
def patched_attention(self, hidden_states, ...):
    q, k, v = self.compute_qkv(hidden_states)
    
    # Apply sparse adapter
    sparse_out = sparse_adapters[layer_idx](hidden_states, q, k, v)
    
    return self.output_proj(sparse_out)
```

### 2. No CUDA Kernels

**Status**: ⬜ Not implemented

**Impact**: 
- Current implementation may not be faster
- Gather operations are slow in PyTorch
- No kernel fusion

**Solution**: Implement Triton kernels (see Native Sparse Attention reference)

### 3. No KV Cache Support

**Status**: ⬜ Not implemented

**Impact**: Cannot efficiently generate long sequences

**Solution**: Implement incremental generation with cached compressed KV

## 🔮 Future Work

### Priority 0 (Essential)
- [ ] Forward hook integration ⚡ **MOST IMPORTANT**
- [ ] Verify training convergence on small dataset
- [ ] Basic performance benchmark

### Priority 1 (Important)
- [ ] KV cache for generation
- [ ] Comprehensive ablation studies
- [ ] Speed/memory profiling

### Priority 2 (Nice to have)
- [ ] Triton kernel implementation
- [ ] FlexAttention integration
- [ ] Multi-task training

## 📚 References

1. **Native Sparse Attention** (DeepSeek, 2025)
   - Paper: https://arxiv.org/abs/2502.11089
   - Repo: https://github.com/lucidrains/native-sparse-attention-pytorch

2. **SoftCoT** (Efficient Speculative Decoding)
   - Provides training pipeline reference

3. **Grouped Query Attention**
   - Used in Llama 3.2 (num_heads=32, num_kv_heads=8)

## 🤝 Contributing

Priority areas:
1. Forward pass integration implementation
2. CUDA/Triton kernels
3. Performance benchmarking
4. Bug fixes and optimizations

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{cs441_sparse_attention,
  title={Trainable Sparse Attention for LLM Inference Acceleration},
  author={CS441 Project},
  year={2025},
  note={Based on Native Sparse Attention and SoftCoT}
}
```

---

**Status**: ✅ Core architecture implemented, ⬜ Integration pending

**Last Updated**: December 2025

**For detailed documentation, see**:
- `README_SPARSE_ATTENTION.md` - User guide
- `DESIGN_GUIDE_CN.md` - Design details (Chinese)

