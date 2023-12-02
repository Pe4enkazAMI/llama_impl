import sentencepiece as spm
import os


class Tokenizer:
    def __init__(self, model_path="lm.model"):
        self.model_prefix = os.path.splitext(model_path)[0]
        self.model_path = model_path
        self.processor = None
        self.load_processor()

    def train(self, input_file, vocab_size, model_type):
        spm.SentencePieceTrainer.train(
            input=input_file,
            vocab_size=vocab_size,
            model_prefix=self.model_prefix,
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )

    def load_processor(self):
        if os.path.exists(self.model_path):
            self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    def encode(self, text):
        return self.processor.encode(text)
    
    def decode(self, ids):
        return self.processor.decode(ids)
