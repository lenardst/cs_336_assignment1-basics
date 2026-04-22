import multiprocessing as mp
import os

import regex as re

PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pre_tokenize_chunk(texts: list[str], special_tokens: tuple[str, ...]) -> list[str]:
    tokenized_chunk: list[str] = []
    special_tokens_set = set(special_tokens)

    for text in texts:
        if text == "":
            continue
        if text in special_tokens_set:
            tokenized_chunk.append(text)
            continue
        tokenized_chunk.extend(m.group(0) for m in re.finditer(PRETOKEN_PATTERN, text))

    return tokenized_chunk


class BPE_Tokenizer:
    def __init__(self, special_tokens: list[str]):
        self.special_tokens = special_tokens
        self.vocabulary: dict[int, bytes] = self._init_vocab(special_tokens)
        self.merges: list[tuple[bytes, bytes]] = []

    def _init_vocab(self, special_tokens: list[str]) -> dict[int, bytes]:
        vocabulary: dict[int, bytes] = {}

        for b in range(256):
            vocabulary[b] = bytes([b])

        for i, token in enumerate(special_tokens):
            vocabulary[256 + i] = token.encode("utf-8")

        return vocabulary

    def _add_to_vocab(self, word: bytes) -> int:
        new_key = len(self.vocabulary)
        self.vocabulary[new_key] = word
        return new_key

    def _add_merge(self, first_token: bytes, second_token: bytes):
        self.merges.append((first_token, second_token))


    def split_by_special_tokens(self, text: str) -> list[str]:
        pattern = "|".join(re.escape(token) for token in self.special_tokens)
        chunks = re.split(f"({pattern})", text)
        return chunks

    def pre_tokenize(
        self,
        text: str,
        num_processes: int | None = None,
        min_parallel_chunks: int = 1024,
    ) -> list[str]:
        if text == "":
            return []

        texts = self.split_by_special_tokens(text)
        special_tokens = tuple(self.special_tokens)
        if num_processes is None:
            num_processes = os.cpu_count() or 1
        if num_processes <= 1 or len(texts) < min_parallel_chunks:
            return _pre_tokenize_chunk(texts, special_tokens)

        # Build units that only end at special tokens.
        special_tokens_set = set(self.special_tokens)
        units: list[list[str]] = []
        current_unit: list[str] = []
        for token in texts:
            current_unit.append(token)
            if token in special_tokens_set:
                units.append(current_unit)
                current_unit = []
        if current_unit:
            units.append(current_unit)

        if len(units) <= 1:
            return _pre_tokenize_chunk(texts, special_tokens)

        with mp.Pool(processes=min(num_processes, len(units))) as pool:
            chunk_results = pool.starmap(_pre_tokenize_chunk, [(unit, special_tokens) for unit in units])

        pretokenized: list[str] = []
        for chunk_tokens in chunk_results:
            pretokenized.extend(chunk_tokens)
        return pretokenized

    def count_words(self, tokenized_text: list[str]) -> dict[tuple[bytes, ...], int]:
        word_counts: dict[tuple[bytes, ...], int] = {}
        special_tokens_set = set(self.special_tokens)

        for word in tokenized_text:
            # Keep special tokens atomic so BPE never merges inside them.
            if word in special_tokens_set:
                word_tuple = (word.encode("utf-8"),)
            else:
                word_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
            if word_tuple in word_counts:
                word_counts[word_tuple] += 1
            else:
                word_counts[word_tuple] = 1
        return word_counts

    def initial_pair_counts(
        self, pre_tokenized_counts: dict[tuple[bytes, ...], int]
    ) -> dict[tuple[bytes, bytes], int]:
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        for word_tuple, count in pre_tokenized_counts.items():
            tuple_size = len(word_tuple)
            if tuple_size > 1:
                for start_position in range(tuple_size - 1):
                    first_char = word_tuple[start_position]
                    second_char = word_tuple[start_position + 1]
                    pair_tuple = (first_char, second_char)
                    if pair_tuple in pair_counts:
                        pair_counts[pair_tuple] += count
                    else:
                        pair_counts[pair_tuple] = count
        return pair_counts

    def merge_tokens_words_element(
        self,
        to_merge: tuple[bytes, bytes],
        tokenized_counts: dict[tuple[bytes, ...], int],
    ) -> dict[tuple[bytes, ...], int]:
        first, second = to_merge
        new_tokenized_counts = tokenized_counts.copy()
        for word_tuple in tokenized_counts:
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if (
                    i < len(word_tuple) - 1
                    and word_tuple[i] == first
                    and word_tuple[i + 1] == second
                ):
                    new_tuple.append(first + second)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_tokenized_counts[tuple(new_tuple)] = new_tokenized_counts.pop(word_tuple)
        return new_tokenized_counts

    def is_subsequence(self, sub: tuple[bytes, ...], main: tuple[bytes, ...]) -> bool:
        n, m = len(sub), len(main)
        return any(main[i : i + n] == sub for i in range(m - n + 1))

    def build_pair_index(
        self, tokenized_counts: dict[tuple[bytes, ...], int]
    ) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
        pair_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
        for word_tuple in tokenized_counts:
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                if pair not in pair_index:
                    pair_index[pair] = set()
                pair_index[pair].add(word_tuple)
        return pair_index

    def apply_merge(
        self,
        best_pair: tuple[bytes, bytes],
        tokenized_counts: dict[tuple[bytes, ...], int],
        pair_counts: dict[tuple[bytes, bytes], int],
        pair_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    ) -> None:
        first, second = best_pair
        merged = first + second
        affected_words = list(pair_index.get(best_pair, set()))

        for word_tuple in affected_words:
            count = tokenized_counts[word_tuple]

            # Step 1: remove stale pairs (old word)
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) - count
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                if pair in pair_index:
                    pair_index[pair].discard(word_tuple)

            # Step 2: apply merge to produce new word
            new_list = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_list.append(merged)
                    i += 2
                else:
                    new_list.append(word_tuple[i])
                    i += 1
            new_word_tuple = tuple(new_list)

            # Step 3: update tokenized_counts
            tokenized_counts[new_word_tuple] = tokenized_counts.pop(word_tuple)

            # Step 4: add new pairs (new word)
            for i in range(len(new_word_tuple) - 1):
                pair = (new_word_tuple[i], new_word_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                if pair not in pair_index:
                    pair_index[pair] = set()
                pair_index[pair].add(new_word_tuple)

        pair_index.pop(best_pair, None)

    def update_word_pairs(
        self,
        to_merge: tuple[bytes, bytes],
        pair_counts: dict[tuple[bytes, bytes], int],
        tokenized_counts: dict[tuple[bytes, ...], int],
    ) -> dict[tuple[bytes, bytes], int]:
        first, second = to_merge
        affected_tokens = {first, second, first + second}

        # Keep unaffected pairs as-is.
        new_pair_counts = {
            pair: count
            for pair, count in pair_counts.items()
            if pair[0] not in affected_tokens and pair[1] not in affected_tokens
        }

        # Recompute only pairs that touch first/second/merged.
        for word_tuple, count in tokenized_counts.items():
            if len(word_tuple) < 2:
                continue
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                if pair[0] in affected_tokens or pair[1] in affected_tokens:
                    new_pair_counts[pair] = new_pair_counts.get(pair, 0) + count

        return new_pair_counts

    def train(self, file_path: str, max_vocab_size: int):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        pre_tokenized_text = self.pre_tokenize(text)
        pre_tokenized_counts = self.count_words(pre_tokenized_text)
        token_counts = pre_tokenized_counts
        pair_counts = self.initial_pair_counts(token_counts)
        pair_index = self.build_pair_index(token_counts)
        while len(self.vocabulary) < max_vocab_size:
            best_pair, highest_count = max(
            pair_counts.items(),
                key=lambda item: (item[1], item[0])
            )
            self._add_merge(best_pair[0], best_pair[1])
            new_word = best_pair[0] + best_pair[1]
            # print(f'Merge {new_word}')
            self._add_to_vocab(new_word)
            self.apply_merge(best_pair, token_counts, pair_counts, pair_index)
        print("Vocab size reached.")
    

def train_bpe(input_path, vocab_size, special_tokens):
    bpe_tokenizer = BPE_Tokenizer(special_tokens)
    bpe_tokenizer.train(input_path, vocab_size)
    vocab = bpe_tokenizer.vocabulary
    merges = bpe_tokenizer.merges
    return vocab, merges
