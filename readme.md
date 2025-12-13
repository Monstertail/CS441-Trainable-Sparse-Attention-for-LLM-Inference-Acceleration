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

fine_tune-> (could ignore) To fine tune in multiple GPUs. Currently, I focused on the pretraining.

### launch scripts

#### pretraining

can change sparse attention type
seq length, or other sparse attention settings.

#### evaluation

## 📁 Project Structure

### data collection
I collect some data from UIUC CS441 course when I prepared for the exam. To generate more data, I designed the prompt to let Gemini 3 to help me genrate synthetic data in a question-answer format.  Check details in [`data_collection/readme.md`](data_collection/readme.md).

#### Usage in prertaining
I pretrained the transformer with enwiki dataset. 


#### Usage in continuous-pretraining(aka middle training)
I had planed to train the model with a train set of my collected CS441 knowlegde dataset after pretraining with enwiki dataset. I did not have enough time to do so, but it does not matter as we can still use the data to evaluate the pretrained model with an out-of-distribution simulation.

#### Usage in evaluation
During the evaluation, I used a subset of enwiki to simulate the in-distribution case, and used my collected data of CS441 to simulate the out-of-distribution case.

### Model design
<div align="center">
  <img src="assets/cs441_nsa.png" />
</div>

The oraginal [native trainable sparse attention(NSA)](https://arxiv.org/pdf/2502.11089 ) from DeepSeek has three components: (1) compression module for global information; (2) fine-grained block selection for fine-grained middle information; (3) sliding window attention for local information. Then all three components' outputs will be combined in a gate to get the attention output.

Compression module is very important as it not only provide the global information, but also it is the base of fine-grained block selection. In the original NSA paper, it adopts a MLP to compress the KV cache. But another block sparse attention paper from Moonshot AI called Mixture of Block Attention(MoBA) mentioned that the meanpooling of KV cache is already enough to do the compression.

Therefore, I implemented and compared four kinds of compression methods, as shown in the figure above. I summaried the details of those four methods here:


### Evaluation

Different compression modules and max context length
#### pretraining time observation
training loss and evaluation loss

#### Efficiency
metrics: memory, latency, throughput

#### Quality
metrics: Perplexity in in-distribution and out-of-distribution case.





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





