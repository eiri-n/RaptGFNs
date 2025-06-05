"""
DNAアプタマー生成用ユーティリティ関数とクラス
"""

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from collections.abc import MutableMapping
from cachetools import cached, LRUCache

# DNAアプタマー用塩基
APTAMER_BASES = ["A", "C", "T", "G"]
APTAMER_ALPHABET = ["[PAD]", "[CLS]", "[UNK]", "[MASK]", "[SEP]"] + APTAMER_BASES + ["0"]


def padding_collate_fn(batch, padding_value=0.0):
    """バッチデータのパディング処理"""
    with torch.no_grad():
        if isinstance(batch[0], tuple):
            k = len(batch[0])
            x = torch.nn.utils.rnn.pad_sequence(
                [b[0] for b in batch], batch_first=True, padding_value=padding_value
            )
            rest = [torch.stack([b[i] for b in batch]) for i in range(1, k)]
            return (x,) + tuple(rest)
        else:
            x = torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=padding_value
            )
            return x


def flatten_config(d, parent_key='', sep='_'):
    """設定辞書の平坦化"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class IntTokenizer:
    """整数ベースのトークナイザー基底クラス"""
    
    def __init__(self, non_special_vocab, full_vocab, padding_token="[PAD]",
                 masking_token="[MASK]", bos_token="[CLS]", eos_token="[SEP]"):
        self.non_special_vocab = non_special_vocab
        self.full_vocab = full_vocab
        self.special_vocab = set(full_vocab) - set(non_special_vocab)
        self.lookup = {a: i for (i, a) in enumerate(full_vocab)}
        self.inverse_lookup = {i: a for (i, a) in enumerate(full_vocab)}
        self.padding_idx = self.lookup[padding_token]
        self.masking_idx = self.lookup[masking_token]
        self.bos_idx = self.lookup[bos_token]
        self.eos_idx = self.lookup[eos_token]

        self.sampling_vocab = non_special_vocab
        self.non_special_idxs = [self.convert_token_to_id(t) for t in non_special_vocab]
        self.special_idxs = [self.convert_token_to_id(t) for t in self.special_vocab]

    @cached(cache=LRUCache(maxsize=int(1e4)))
    def encode(self, seq, use_sep=True):
        """配列をトークンIDにエンコード"""
        if seq.endswith("%"):
            seq = ["[CLS]"] + list(seq[:-1])
            seq += ["[SEP]"] if use_sep else []
            return [self.convert_token_to_id(c) for c in seq] + [4]
        else:
            seq = ["[CLS]"] + list(seq)
            seq += ["[SEP]"] if use_sep else []
            return [self.convert_token_to_id(c) for c in seq]

    def decode(self, token_ids):
        """トークンIDを配列にデコード"""
        if isinstance(token_ids, int):
            return self.convert_id_to_token(token_ids)

        tokens = []
        for t_id in token_ids:
            token = self.convert_id_to_token(t_id)
            if token in self.special_vocab and token not in ["[MASK]", "[UNK]"]:
                continue
            tokens.append(token)
        return ' '.join(tokens)

    def convert_id_to_token(self, token_id):
        """トークンIDをトークンに変換"""
        if torch.is_tensor(token_id):
            token_id = token_id.item()
        assert isinstance(token_id, int)
        return self.inverse_lookup.get(token_id, '[UNK]')

    def convert_token_to_id(self, token):
        """トークンをトークンIDに変換"""
        unk_idx = self.lookup["[UNK]"]
        return self.lookup.get(token, unk_idx)

    def set_sampling_vocab(self, sampling_vocab=None, max_ngram_size=1):
        """サンプリング用語彙の設定"""
        if sampling_vocab is None:
            sampling_vocab = []
            import itertools
            for i in range(1, max_ngram_size + 1):
                prod_space = [self.non_special_vocab] * i
                for comb in itertools.product(*prod_space):
                    sampling_vocab.append("".join(comb))
        else:
            new_tokens = set(sampling_vocab) - set(self.full_vocab)
            self.full_vocab.extend(list(new_tokens))
            self.lookup = {a: i for (i, a) in enumerate(self.full_vocab)}
            self.inverse_lookup = {i: a for (i, a) in enumerate(self.full_vocab)}

        self.sampling_vocab = sampling_vocab


class AptamerTokenizer(IntTokenizer):
    """DNAアプタマー専用トークナイザー"""
    
    def __init__(self):
        super().__init__(APTAMER_BASES, APTAMER_ALPHABET)


def random_aptamer_strings(num, min_len=30, max_len=60):
    """ランダムなDNAアプタマー配列を生成"""
    strs = []
    for _ in range(num):
        length = np.random.randint(min_len, max_len + 1)
        idx = np.random.choice(len(APTAMER_BASES), size=length, replace=True)
        strs.append("".join([APTAMER_BASES[i] for i in idx]))
    return np.array(strs)


def str_to_tokens(str_array, tokenizer, use_sep=True):
    """文字列配列をトークン配列に変換"""
    tokens = [
        torch.tensor(tokenizer.encode(x, use_sep)) for x in str_array
    ]
    batch = padding_collate_fn(tokens, tokenizer.padding_idx)
    return batch


def tokens_to_str(tok_idx_array, tokenizer):
    """トークン配列を文字列配列に変換"""
    str_array = np.array([
        tokenizer.decode(token_ids).replace(' ', '') for token_ids in tok_idx_array
    ])
    return str_array


def validate_dna_sequence(sequence):
    """DNA配列の妥当性をチェック"""
    valid_bases = set('ATCG')
    return all(base in valid_bases for base in sequence.upper())


def gc_content(sequence):
    """GC含量を計算"""
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0
