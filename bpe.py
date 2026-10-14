from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=12000,
    special_tokens=["<pad>", "<start>", "<end>", "<unk>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train(["./data/train.txt"], trainer)
tok.save("tinyt_tokenizer.json")
