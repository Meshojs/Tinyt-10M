<img width="1408" height="480" alt="tinytbanner" src="https://github.com/user-attachments/assets/928fd2f4-5dcf-4016-8fbf-f5d0f3160590" />

# Abstract 🧾

<p align="center">
  
We introduce **TinyT-5M** , a 5-million-parameter, decoder-only Transformer trained from scratch on the `roneneldan/TinyStories` dataset.
The model uses a **Byte Pair Encoding (BPE)** tokenizer with a vocabulary size of 12,000 tokens and positional encodings to represent token positions. Its architecture consists of **four Transformer blocks**, each using multi-head self-attention, RMSNorm, GELU activation, and a feed-forward network with a 4× expansion ratio.
We adopt a **pre-norm architecture** with RMSNorm to support training stability and gradient flow. Linear weights are initialized from a normal distribution with a mean of 0 and a standard deviation of 0.02. We configure the **AdamW optimizer** with separate parameter groups, applying weight decay to matrix parameters while excluding one-dimensional parameters, such as biases and RMSNorm scales, from weight decay.
Training uses a learning rate of 3e-4, a batch size of 32, and a cosine learning rate schedule with 300 warmup steps and a minimum learning rate of 3e-5. The model is trained for one epoch, with a maximum of 12,208 training steps.
  
</p>


# Model card 🪪

#### **0.1 - TinyT Config**

```python
model = TinyT(
    vocab_size=12000,
    d_model=192,
    n_blocks=4,
    seq_len=256,
    n_heads=4,
    d_ff=768,
    dropout=0.0,
)
```

#### **0.2 - Model Details**

    Architecture:        Decoder-only Transformer
    Parameters:          ~5M
    Dataset:             roneneldan/TinyStories
    Tokenizer:           BPE
    Vocabulary size:     12,000
    Context length:      256 tokens
    Transformer blocks:  4
    Attention heads:     4
    Hidden dimension:    192
    Feed-forward dim:    768
    Expansion ratio:     4x
    Activation:          GELU
    Normalization:       RMSNorm (Pre-Norm)
    Optimizer:           AdamW
    Dropout:             0.0

#### **0.3 - Training Configuration**

    Epochs:              1
    Maximum steps:       12,208
    Batch size:          32
    Learning rate:       3e-4
    Minimum LR:          3e-5
    Warmup steps:        300
    Weight decay:        0.1
    AdamW betas:         (0.9, 0.95)
    precisions:          mixed precisions bf16


# Generation Test ✍️


<img width="900" alt="BEST - Temperature -0.8"
src="https://github.com/user-attachments/assets/613c9d59-2228-47ea-81cb-9b764824d2c6" />

<img width="900" alt="Temperature -0.5"
src="https://github.com/user-attachments/assets/87460be6-c246-47a8-b8be-9240acabf0f7" />

<img width="900" alt="Temperature -0.8"
src="https://github.com/user-attachments/assets/9c2db908-f3e4-4d0a-9e8b-45435666899f" />
</br>


# Training Condition 📚

     The model's training process was really smooth, as shown in the image above
     loss went from 9.6 to 2.2 ~ 2.4 in 12K steps. The model was trained 
     for 3.30 hours and was trained on 3060 laptop GPU 6 V-rams with mixed precisions bf16


<img width="684" height="31" alt="final" src="https://github.com/user-attachments/assets/4dd941c1-304d-49f2-891e-99aa5f847053" />


</hr>

</br>
</br>
</br>

<div align="center">

**Created with ❤️ by Mashary, of course haha. Thank you!**

</div>
