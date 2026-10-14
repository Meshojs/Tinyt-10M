<img width="1408" height="480" alt="tinytbanner" src="https://github.com/user-attachments/assets/928fd2f4-5dcf-4016-8fbf-f5d0f3160590" />


# Abstract 🧾
<div align="left">
  
We introduce **TinyT-5M**, a 5M-parameter, decoder-only Transformer designed for training on the `roneneldan/TinyStories` dataset.
The model uses a **Byte Pair Encoding (BPE)** tokenizer with a vocabulary size of 12,000 tokens and positional encodings to represent token positions. 
Its architecture consists of five Transformer blocks, each using **multi-head self-attention**, RMSNorm, GELU activation, and a feed-forward network.
We adopt a **pre-norm architecture** with RMSNorm to support training stability and gradient flow. We initialize linear and embedding weights from a
normal distribution. with a mean of 0 and a standard deviation of 0.02. We configure the **AdamW optimizer** with separate parameter groups, applying 
weight decay to matrix parameters while excluding one-dimensional parameters, such as biases and RMSNorm scales, from weight decay.
  
</div>
</hr>
</br>

# Model card 🪪

#### **0.1 - TinyT Config** ####
```python
model = TinyT(
    vocab_size=12000,
    d_model=   256,
    n_blocks=  5,
    seq_len=   256,
    n_heads=   4,
    d_ff=      1024,
    dropout=   0.1,
)

```
#### **0.2 - Model Details** #### 

    Architecture:        Decoder-only Transformer
    Parameters:         ~10M
    Tokenizer:           BPE
    Vocabulary size:     12,000
    Context length:      256 tokens
    Transformer blocks:  5
    Attention heads:     4
    Hidden dimension:    256
    Activation:          GELU
    Normalization:       RMSNorm with pre-norm
    Optimizer:           AdamW
    ratio :              4x ratio
