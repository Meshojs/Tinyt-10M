
import torch
from tokenizers import Tokenizer
from model import TinyT

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file("tinyt_tokenizer.json")

model = TinyT(
    vocab_size=12000,
    d_model=192,
    n_blocks=4,
    seq_len=256,
    n_heads=4,
    d_ff=768,
    dropout=0.0,
).to(device)

model.load_state_dict(
    torch.load("./checkpoints/checkpoint-12000.pt", map_location=device, weights_only=True)
)
model.eval()


@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(x[:, -256:])[:, -1, :]
        logits = logits / temperature

        if top_k:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        x = torch.cat((x, next_token), dim=1)

    return tokenizer.decode(x[0].tolist())



prompt = "Once upon a time, there was a little girl named Lily."

for temperature in [0.5, 0.7, 1.0]:
    print(f"\n{'=' * 40}")
    print(f"Temperature: {temperature}")
    print(generate(
        prompt,
        max_new_tokens=100,
        temperature=temperature,
        top_k=50,
    ))
