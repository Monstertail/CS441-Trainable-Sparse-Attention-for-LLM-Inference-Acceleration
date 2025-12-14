# CS441 Trainable Sparse Attention - Project Summary

## 🎯 Project Overview
Sparse attention is a promising method to reduce the cost during the LLM inference. DeepSeek's native sparse attention is a pretraining method that enables the model to select a subset of KV cache during the inference time, as shown in the follwing figure. In this project, I'd like to show a proof of concept that a GPT-like transformer can be pretrained  to have sparse attention ability.

<div align="center">
  <img src="assets/sparse_overview.jpg" />
</div>

> NOTE: I tried to fine-tune the Llama 3.2-1B to enable sparse attention.
> I tried a lot of efforts to distill the full attention knowledge to sparse attention. But it cannot work well because the fine tuning was not stable.
> I think the reason is due to a small batch size to be within the memory limit of NVIDIA L40.
> That's why I decided to train a transformer from scratch as a proof of concept.
### 🔧 Environment Setup


```bash
conda create -n sparse-attn python=3.10 -y
conda activate sparse-attn
pip install -r requirements.txt
```

### Folder structure

- **pretrain/**: Pretrain a small GPT-like Transformer on enwik8 (byte-level LM).
  - `train.py`: main training script (edit `SEQ_LEN`, `USE_SPARSE_ATTN`, `COMPRESS_METHOD`, etc.)
  - `data/enwik8.gz`: training data (byte stream)
  - `ckpt/`: saved checkpoints
  - `wandb/`: W&B logs

- **sparse_attention/**: Sparse attention related code.
  - `native_sparse_attention_pytorch/`: in-repo copy of DeepSeek Native Sparse Attention (NSA)
    - includes `Transformer`, `SparseAttention`, and `compress_networks.py`
  - additional model wrappers used in this project (e.g., Llama + adapters experiments)

- **evaluation/**: Evaluation scripts (efficiency + quality).
  - `efficiency.py`: latency / throughput / peak memory test
  - `perplexity.py`: PPL test (in-distribution vs out-of-distribution)
  - `pretrain/run_pretrain_efficiency_test.sh`: wrapper to compare checkpoints
  - `pretrain/run_pretrain_ppl_test.sh`: wrapper to compare checkpoints

- **data_collection/**: CS441 QA data (synthetic train/test) used as OOD evaluation source.

- **continuous_pretrain/**: (planned) “middle training / continuous pretraining” on CS441 dataset.

- **fine_tune/**: (to make it work,ignore now) Llama fine-tuning experiments; requires multi-gpus.

### launch scripts

#### pretraining

Pretrain from scratch on enwik8:

```bash
cd pretrain
python train.py
```

Key knobs in `pretrain/train.py`:

- **Attention type**
  - `USE_SPARSE_ATTN = True`: train with NSA sparse attention
  - `USE_SPARSE_ATTN = False`: train a full-attention baseline
- **Sequence length / batch size**: `SEQ_LEN`, `BATCH_SIZE`
- **Compression method (NSA)**
  - `COMPRESS_METHOD = 'conv' | 'attn' | 'mlp' | 'mean'`

#### evaluation

This repo provides two evaluation dimensions:

- **Efficiency (speed + memory)**:

```bash
# from repo root
bash evaluation/pretrain/run_pretrain_efficiency_test.sh 5000
```

It loads checkpoints under `pretrain/ckpt/` and prints latency / throughput / peak memory.

- **Quality (perplexity, ID vs OOD)**:

```bash
# from repo root
bash evaluation/pretrain/run_pretrain_ppl_test.sh 5000 512
```

It evaluates:
- **In-distribution**: enwik8 validation split (byte-level)
- **Out-of-distribution**: CS441 synthetic QA test set, serialized to UTF-8 bytes

## 📁 Project Structure

### data collection
<div align="center">
  <img src="assets/data_collection.png" />
</div>

I collected some QA-style notes from UIUC CS441 while preparing for the exam. To scale up the data, I designed prompts to let Gemini generate additional synthetic data in a question–answer format. See [`data_collection/readme.md`](data_collection/readme.md).

#### Usage in prertaining
I pretrained the Transformer on **enwik8** as a byte-level language modeling task.


#### Usage in continuous-pretraining(aka middle training)
I planned to run a second-stage “continuous pretraining” on the CS441 dataset after enwik8 pretraining. Due to time/compute constraints it was not fully executed, but the dataset is still useful as an **OOD evaluation** source.

#### Usage in evaluation
During evaluation:
- **In-distribution**: a subset of enwik8 validation (byte-level)
- **Out-of-distribution**: CS441 synthetic QA test set (converted to a byte stream)

### Model design
<div align="center">
  <img src="assets/cs441_nsa.png" />
</div>

The original [Native Sparse Attention (NSA)](https://arxiv.org/pdf/2502.11089) from DeepSeek has three components:
1. **Compression module** for global information (compress long KV into fewer “memory” slots)
2. **Fine-grained block selection** for mid-range information (select a few important remote blocks)
3. **Sliding window attention** for local information

The outputs of all three branches are combined by a learned gate to form the final attention output.

The compression module is critical: it provides global context and also drives fine-grained selection. The original NSA paper uses an MLP-style compressor, while MoBA suggests simple mean pooling can already work surprisingly well.

Therefore, I implemented and compared four compression methods (see `sparse_attention/native_sparse_attention_pytorch/compress_networks.py`):

- **`ConvLinearCompress` (`conv`)**: grouped 1D convolution per head (learned downsampling).
- **`AttentionPool` (`attn`)**: attention-based pooling inside each window (learned weighted average).
- **`MeanPoolCompress` (`mean`)**: parameter-free mean pooling baseline.
- **`GroupedMLP` (`mlp`)**: per-head MLP over the flattened window (higher capacity, higher cost).

<details>
<summary><b>📊Details and key decisions in the model design and training pipeline</b></summary>

I built an end-to-end pipeline to **train**, **evaluate**, and **compare** multiple key design decisions around sparse attention.

#### 1. Models tested (what I compared)

1. **Full-attention baseline (byte-level Transformer)**
   - Same GPT-like architecture and training setup, but with standard full attention as a reference point.

2. **Native Sparse Attention (NSA) Transformer (byte-level)**
   - Same backbone, but replaces attention with the NSA three-branch mechanism (compression + fine selection + sliding window) and a learned gating combiner.

3. **Compression-module variants (within NSA)**
   - Compared multiple NSA variants by changing `COMPRESS_METHOD` while keeping everything else matched.
   - (Details of the four compression networks are summarized above; not repeated here.)

4. **Fine-tuning attempt (Llama + sparse attention adapters, exploratory / partially successful)**
   - Implemented a model wrapper to plug trainable NSA adapters into a frozen Llama teacher.
   - This line of work was valuable for learning, but was unstable under strict memory limits (small effective batch sizes), so the main results focus on pretraining from scratch.

#### 2. Hyperparameters and key design knobs I varied

1. **Context length**
   - Tested different `SEQ_LEN` (e.g., 512 and longer contexts) to study scaling behavior and efficiency trade-offs.

2. **Attention mode**
   - `USE_SPARSE_ATTN = True` (NSA) vs `USE_SPARSE_ATTN = False` (full attention baseline).

3. **NSA sparsity structure**
   - Sliding window and selection behavior: `SLIDING_WINDOW_SIZE`, `FINE_BLOCK_SIZE`, `NUM_FINE_SELECTED`
   - Compression windowing: `COMPRESS_BLOCK_SIZE`, `COMPRESS_BLOCK_SLIDING_STRIDE`
   - Selection behavior toggles: `USE_DIFF_TOPK`, `QUERY_HEADS_SHARE_SELECTION`
   - Kernel choice for fine selection (where supported): triton / flex / vanilla paths

4. **Training pipeline knobs**
   - Optimization and stability: learning rate, gradient accumulation, gradient clipping, checkpointing cadence, and early stopping threshold.

#### 3. Data + preprocessing (byte-level throughout)

1. **In-distribution (ID)**
   - enwik8 byte stream (no tokenizer; vocab size = 256).

2. **Out-of-distribution (OOD)**
   - CS441 synthetic QA test set.
   - Since there is no encoder/tokenizer, I serialize QA pairs into text and convert to **UTF-8 bytes** so it can be evaluated in the same byte-level LM setting.

#### 4. Evaluation pipeline (what I implemented to compare decisions)

1. **Efficiency**
   - Measured peak GPU memory, latency, and throughput using a dedicated script and wrapper.

2. **Quality**
   - Measured perplexity (PPL) on both ID (enwik8 val) and OOD (CS441 QA bytes), and printed the evaluated token / example counts for transparency.

#### 5. Fine-tuning / distillation losses explored (Llama adapter attempt)

To bridge full attention → sparse attention, I experimented with multiple loss designs beyond plain CE:
- **CE loss** for next-token prediction (standard LM objective)
- **KL distillation on logits** (teacher vs student output distribution)
- **Layer-wise MSE on hidden states** (align intermediate representations)
- **Mixed objectives** combining distillation + CE

These experiments guided the project direction: due to instability under memory constraints, the most reliable results came from the from-scratch pretraining pipeline + systematic ablations.

</details>

### Evaluation
Summarize the metrics for different evaluations here:

#### training time observation
- training loss and evaluation loss (enwik8 val)



##### Pretrain( train a small transformer from scratch)
<div align="center">
  <img src="assets/pretrain_loss.jpg" />
  <figcaption> Loss for a transformer pretrained with enwik8 dataset in full/sparse attention. Max sequence length=4096 and training steps=5000.</figcaption>
</div>

We can see sparse attention with compresion module of `GroupedMLP` can be even better than full attention: it converges faster because it can focus the important information more accurately than other compression methods and filter those noisy tokens in full attention.

##### Fine-tune( based on Llama 3.2-12B)
<figure style="display:flex; gap:16px; justify-content:center;">
  <div style="text-align:center;">
    <img src="assets/ft_ce_loss.png" width="40%">
    <figcaption>(left) CE loss during fine-tuning.</figcaption>
  </div>
  <div style="text-align:center;">
    <img src="assets/ft_mse_loss.png" width="40%">
    <figcaption>(right) Average layerwise MSE loss during fine-tuning.</figcaption>
  </div>
</figure>

We fine-tune a **sparse-attention student model** using an **adapter whose input is the original QKV representations**, with a **full-attention LLM as the teacher**.  
We experimented with multiple objectives, including CE, KL, layer-wise MSE, and mixed losses. Empirically, **layer-wise MSE on hidden states leads to more stable and lower training loss than CE**, as shown in the figures.

MSE outperforms CE because:
- It provides denser supervision
- It aligns representations instead of predictions
- It is more robust to architectural mismatch (full → sparse attention)
- It synergizes well with a QKV-based adapter

But:
- MSE is still data- and batch-hungry
- Stability improves with larger batches, better normalization, or staged training

#### Efficiency
- metrics: decoding throughput, KV cache memory access saving among different batch sizes and prompt lengths.

#### Quality
- metrics: Perplexity in in-distribution and out-of-distribution case (enwik8 vs CS441)

<div align="center">
  <img src="assets/ppl_bars_step5000.png" />
  <figcaption> Perplexity in in-distribution and out-of-distribution case (enwik8 vs CS441) for different attention methods and sequence lengthes.</figcaption>
</div>

We can see the inference time evaluation is quite aligned with the training behavior among those attention methods:
- sparse attention+ `GroupedMLP` works best among not only all sparse attention methods but also be better than full attention, in both in-distrubtion and out-of-distribution cases.;
- sparse attention+ `MeanPoolCompress` is the worse in perplexity because it's parameter-free.
- In-distribution evaluation has lower perplexity than out-of-distribution case.


## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{cs441_sparse_attention,
  title={Pretrain transformer(LLM) with sparse attention},
  author={Jinwei Yao},
  year={2025},
  
}
```

## Acknowledge





