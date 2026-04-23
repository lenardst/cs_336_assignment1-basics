import pickle
from typing import Iterable, Iterator
from importlib import import_module

pre_tokenize = import_module("cs336_basics.2_0_bpe_tokenizer").pre_tokenize

def read_with_sequence_boundary(iterable: Iterable[str], delimiters):
    remainder = ""
    for chunk in iterable:
        current_data = remainder + chunk
        best_split_index = -1
        best_delim_len = 0
        for delim in delimiters:
            idx = current_data.rfind(delim)
            if idx > best_split_index:
                best_split_index = idx
                best_delim_len = len(delim)
        if best_split_index != -1:
            split_point = best_split_index + best_delim_len
            yield current_data[:split_point]
            remainder = current_data[split_point:]
        else:
            remainder = current_data
    if remainder:
        yield remainder

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self._special_tokens_set = set(self.special_tokens)
        self._special_token_bytes = {token: token.encode("utf-8") for token in self.special_tokens}
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self._merge_ranks = {(first, second): rank for rank, (first, second) in enumerate(self.merges)}
        self._piece_cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)
        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)
        if special_tokens:
            for special_token in special_tokens:
                encoded_special_token = special_token.encode("utf-8")
                if encoded_special_token not in vocab.values():
                    key =  max(vocab.keys()) + 1
                    vocab[key] = encoded_special_token
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokenized = pre_tokenize(text, self.special_tokens)
        encoded_ids: list[int] = []
        piece_cache = self._piece_cache
        token_to_id = self.token_to_id
        merge_ranks = self._merge_ranks
        special_tokens_set = self._special_tokens_set
        special_token_bytes = self._special_token_bytes

        for piece in pretokenized:
            cached = piece_cache.get(piece)
            if cached is not None:
                encoded_ids.extend(cached)
                continue

            if piece in special_tokens_set:
                token_bytes = [special_token_bytes[piece]]
            else:
                token_bytes = [bytes([b]) for b in piece.encode("utf-8")]

            while len(token_bytes) > 1:
                best_pair: tuple[bytes, bytes] | None = None
                best_rank: int | None = None
                for i in range(len(token_bytes) - 1):
                    pair = (token_bytes[i], token_bytes[i + 1])
                    rank = merge_ranks.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_pair = pair

                if best_pair is None:
                    break
                first, second = best_pair
                i = 0
                merged_tokens: list[bytes] = []
                while i < len(token_bytes):
                    if (
                        i + 1 < len(token_bytes)
                        and token_bytes[i] == first
                        and token_bytes[i + 1] == second
                    ):
                        merged_tokens.append(first + second)
                        i += 2
                    else:
                        merged_tokens.append(token_bytes[i])
                        i += 1
                token_bytes = merged_tokens

            piece_ids: list[int] = []
            for token in token_bytes:
                token_id = token_to_id.get(token)
                if token_id is None:
                    raise ValueError(f"Token {token!r} not found in vocabulary.")
                piece_ids.append(token_id)
            encoded_ids.extend(piece_ids)
            piece_cache[piece] = tuple(piece_ids)
        return encoded_ids

    def encode_iterable(self, iterable:Iterable[str]) -> Iterator[int]:
        for chunk in read_with_sequence_boundary(iterable, delimiters=self.special_tokens):
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        bytes_out = b"".join(self.vocab[token_id] for token_id in ids)
        return bytes_out.decode("utf-8", errors="replace")
