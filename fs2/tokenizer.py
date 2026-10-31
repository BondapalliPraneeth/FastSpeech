import json
import os


class Tokenizer:
    """Maps a "{p1 p2 ...}" phone string to ids using symbols.json written by preprocess.py.

    Replaces ming024's `text` package (English-only cleaners + fixed ARPAbet symbol set).
    id 0 is reserved for padding.
    """

    def __init__(self, preprocessed_path):
        with open(os.path.join(preprocessed_path, "symbols.json"), encoding="utf-8") as f:
            self.symbols = json.load(f)
        self.sym2id = {s: i + 1 for i, s in enumerate(self.symbols)}

    @property
    def vocab_size(self):
        return len(self.symbols) + 1

    def encode(self, phone_string):
        phones = phone_string.strip().strip("{}").split()
        return [self.sym2id[p] for p in phones if p in self.sym2id]
