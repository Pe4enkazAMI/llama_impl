import sentencepiece as sps

class Tokenizer:
    def __init__(self, path_to_corpora, vocab_size, model_prefix, model_type) -> None:
        self.path_to_corpora = path_to_corpora
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.model_type = model_type

    def fit(self):
        string = f"--input={self.path_to_corpora} --model_prefix={self.model_prefix} --vocab_size={self.vocab_size} --model_type={self.model_type} --user_defined_symbols=[PAD]"
        sps.SentencePieceTrainer.train(string)
        self.tokenizer = sps.SentencePieceProcessor()
        self.tokenizer.load(f"{self.model_prefix}.model")

    def load(self, path):
        self.tokenizer = sps.SentencePieceProcessor()
        self.tokenizer.load(f"{path}.model")
        
    def encode(self, string):
        return self.tokenizer.Encode(string)
    
    def decode(self, ids):
        return self.tokenizer.DecodeIds(ids)
