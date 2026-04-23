import multiprocessing as mp
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import chain

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


def pre_tokenize(
    text: str,
    special_tokens: list[str],
    num_processes: int | None = None,
    min_parallel_chunks: int = 1024,
) -> list[str]:
    if text == "":
        return []

    if not special_tokens:
        texts = [text]
    else:
        pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        texts = re.split(f"({pattern})", text)
    special_tokens_tuple = tuple(special_tokens)
    if num_processes is None:
        num_processes = os.cpu_count() or 1
    if num_processes <= 1 or len(texts) < min_parallel_chunks:
        return _pre_tokenize_chunk(texts, special_tokens_tuple)

    # Build units that only end at special tokens.
    special_tokens_set = set(special_tokens)
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
        return _pre_tokenize_chunk(texts, special_tokens_tuple)

    with mp.Pool(processes=min(num_processes, len(units))) as pool:
        chunk_results = pool.starmap(
            _pre_tokenize_chunk, [(unit, special_tokens_tuple) for unit in units]
        )

    pretokenized: list[str] = []
    for chunk_tokens in chunk_results:
        pretokenized.extend(chunk_tokens)
    return pretokenized


def _count_words_in_parts_batch(
    args: tuple[list[str], tuple[str, ...], dict[str, int]],
) -> dict[tuple[int, ...], int]:
    parts, special_tokens, special_token_to_id = args
    special_tokens_set = set(special_tokens)
    word_counts: dict[tuple[int, ...], int] = {}

    for part in parts:
        if not part:
            continue
        if part in special_tokens_set:
            word_tuple = (special_token_to_id[part],)
            word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1
            continue

        for match in re.finditer(PRETOKEN_PATTERN, part):
            word_tuple = tuple(match.group(0).encode("utf-8"))
            word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1

    return word_counts


def _merge_word_counts_into(
    destination: dict[tuple[int, ...], int],
    source: dict[tuple[int, ...], int],
) -> None:
    for word_tuple, count in source.items():
        destination[word_tuple] = destination.get(word_tuple, 0) + count


class BPE_Tokenizer:
    def __init__(self, special_tokens: list[str]):
        self.special_tokens = special_tokens
        self.vocabulary: dict[int, bytes] = self._init_vocab(special_tokens)
        self.special_token_to_id: dict[str, int] = {
            token: 256 + i for i, token in enumerate(special_tokens)
        }
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
        if not self.special_tokens:
            return [text]
        pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
        chunks = re.split(f"({pattern})", text)
        return chunks

    def count_words(self, tokenized_text: list[str]) -> dict[tuple[int, ...], int]:
        word_counts: dict[tuple[int, ...], int] = {}
        special_tokens_set = set(self.special_tokens)

        for word in tokenized_text:
            # Keep special tokens atomic so BPE never merges inside them.
            if word in special_tokens_set:
                word_tuple = (self.special_token_to_id[word],)
            else:
                word_tuple = tuple(word.encode("utf-8"))
            if word_tuple in word_counts:
                word_counts[word_tuple] += 1
            else:
                word_counts[word_tuple] = 1
        return word_counts

    def _iter_text_parts_by_special_tokens(
        self,
        file_path: str,
        read_chars: int = 4_000_000,
    ):
        """Yield alternating non-special text and matched special-token parts."""
        if not self.special_tokens:
            with open(file_path, encoding="utf-8") as file:
                yield file.read()
            return

        pattern = re.compile(
            "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
        )
        buffer = ""
        with open(file_path, encoding="utf-8") as file:
            while True:
                chunk = file.read(read_chars)
                if not chunk:
                    break
                buffer += chunk
                start = 0
                for match in pattern.finditer(buffer):
                    if match.start() > start:
                        yield buffer[start:match.start()]
                    yield match.group(0)
                    start = match.end()
                buffer = buffer[start:]
            if buffer:
                yield buffer

    def count_words_from_file(self, file_path: str) -> dict[tuple[int, ...], int]:
        """
        Stream pre-tokenization/counting from file to avoid materializing
        the full corpus and pretokenized token list in memory.
        """
        def iter_part_batches(parts_per_batch: int = 4096):
            batch: list[str] = []
            for part in self._iter_text_parts_by_special_tokens(file_path):
                batch.append(part)
                if len(batch) >= parts_per_batch:
                    yield batch
                    batch = []
            if batch:
                yield batch

        word_counts: dict[tuple[int, ...], int] = {}
        num_processes = os.cpu_count() or 1
        min_parallel_batches = 2
        batch_iterator = iter_part_batches()
        prefetched_batches: list[list[str]] = []
        for _ in range(min_parallel_batches):
            try:
                prefetched_batches.append(next(batch_iterator))
            except StopIteration:
                break

        all_batches = chain(prefetched_batches, batch_iterator)
        special_tokens_tuple = tuple(self.special_tokens)

        if num_processes <= 1 or len(prefetched_batches) < min_parallel_batches:
            for parts_batch in all_batches:
                local_counts = _count_words_in_parts_batch(
                    (parts_batch, special_tokens_tuple, self.special_token_to_id)
                )
                _merge_word_counts_into(word_counts, local_counts)
            return word_counts

        with mp.Pool(processes=num_processes) as pool:
            worker_inputs = (
                (parts_batch, special_tokens_tuple, self.special_token_to_id)
                for parts_batch in all_batches
            )
            for local_counts in pool.imap_unordered(_count_words_in_parts_batch, worker_inputs):
                _merge_word_counts_into(word_counts, local_counts)

        return word_counts

    def initial_pair_counts(
        self, pre_tokenized_counts: dict[tuple[int, ...], int]
    ) -> dict[tuple[int, int], int]:
        pair_counts: dict[tuple[int, int], int] = {}
        for word_tuple, count in pre_tokenized_counts.items():
            tuple_size = len(word_tuple)
            if tuple_size > 1:
                for start_position in range(tuple_size - 1):
                    first_char: int = word_tuple[start_position]
                    second_char: int = word_tuple[start_position + 1]
                    pair_tuple = (first_char, second_char)
                    if pair_tuple in pair_counts:
                        pair_counts[pair_tuple] += count
                    else:
                        pair_counts[pair_tuple] = count
        return pair_counts

    def merge_tokens_words_element(
        self,
        to_merge: tuple[int, int],
        tokenized_counts: dict[tuple[int, ...], int],
    ) -> dict[tuple[int, ...], int]:
        first, second = to_merge
        merged_id = len(self.vocabulary)
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
                    new_tuple.append(merged_id)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_tokenized_counts[tuple(new_tuple)] = new_tokenized_counts.pop(word_tuple)
        return new_tokenized_counts

    def is_subsequence(self, sub: tuple[int, ...], main: tuple[int, ...]) -> bool:
        n, m = len(sub), len(main)
        return any(main[i : i + n] == sub for i in range(m - n + 1))

    def build_pair_index(
        self, tokenized_counts: dict[tuple[int, ...], int]
    ) -> dict[tuple[int, int], set[tuple[int, ...]]]:
        pair_index: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        for word_tuple in tokenized_counts:
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                if pair not in pair_index:
                    pair_index[pair] = set()
                pair_index[pair].add(word_tuple)
        return pair_index

    def apply_merge(
        self,
        best_pair: tuple[int, int],
        merged_id: int,
        tokenized_counts: dict[tuple[int, ...], int],
        pair_counts: dict[tuple[int, int], int],
        pair_index: dict[tuple[int, int], set[tuple[int, ...]]],
    ) -> None:
        first, second = best_pair
        affected_words = list(pair_index.get(best_pair, set()))

        for word_tuple in affected_words:
            count = tokenized_counts[word_tuple]

            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) - count
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                if pair in pair_index:
                    pair_index[pair].discard(word_tuple)

            new_list = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_list.append(merged_id)
                    i += 2
                else:
                    new_list.append(word_tuple[i])
                    i += 1
            new_word_tuple = tuple(new_list)

            tokenized_counts[new_word_tuple] = tokenized_counts.pop(word_tuple)

            for i in range(len(new_word_tuple) - 1):
                pair = (new_word_tuple[i], new_word_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                if pair not in pair_index:
                    pair_index[pair] = set()
                pair_index[pair].add(new_word_tuple)

        pair_index.pop(best_pair, None)

    def update_word_pairs(
        self,
        to_merge: tuple[int, int],
        pair_counts: dict[tuple[int, int], int],
        tokenized_counts: dict[tuple[int, ...], int],
    ) -> dict[tuple[int, int], int]:
        first, second = to_merge
        merged_id = len(self.vocabulary)
        affected_tokens = {first, second, merged_id}

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

    def train(
        self,
        file_path: str,
        max_vocab_size: int,
        progress_every: int | None = None,
        progress_interval_seconds: float | None = None,
        include_timestamps: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ):
        token_counts = self.count_words_from_file(file_path)
        pair_counts = self.initial_pair_counts(token_counts)
        pair_index = self.build_pair_index(token_counts)
        start_time = time.perf_counter()
        initial_vocab_size = len(self.vocabulary)
        target_merges = max(0, max_vocab_size - initial_vocab_size)

        def emit_progress(message: str) -> None:
            if include_timestamps:
                timestamp = datetime.now(UTC).isoformat(timespec="seconds")
                message = f"[{timestamp}] {message}"
            if progress_callback is not None:
                progress_callback(message)
            else:
                print(message)

        emit_progress(
            f"[train] start vocab={initial_vocab_size} target_vocab={max_vocab_size} "
            f"target_merges={target_merges}"
        )
        last_progress_log_time = start_time
        while len(self.vocabulary) < max_vocab_size:
            best_pair, highest_count = max(
                pair_counts.items(),
                key=lambda item: (
                    item[1],
                    (self.vocabulary[item[0][0]], self.vocabulary[item[0][1]]),
                ),
            )
            first_token = self.vocabulary[best_pair[0]]
            second_token = self.vocabulary[best_pair[1]]
            self._add_merge(first_token, second_token)
            new_word = first_token + second_token
            merged_id = self._add_to_vocab(new_word)
            self.apply_merge(best_pair, merged_id, token_counts, pair_counts, pair_index)
            merges_completed = len(self.vocabulary) - initial_vocab_size
            now = time.perf_counter()
            merge_boundary = (
                progress_every is not None
                and progress_every > 0
                and (merges_completed % progress_every == 0 or merges_completed == target_merges)
            )
            time_boundary = (
                progress_interval_seconds is not None
                and progress_interval_seconds > 0
                and (now - last_progress_log_time >= progress_interval_seconds or merges_completed == target_merges)
            )
            if merge_boundary or time_boundary:
                elapsed_s = now - start_time
                merge_rate = merges_completed / elapsed_s if elapsed_s > 0 else 0.0
                remaining_merges = max(0, target_merges - merges_completed)
                eta_s = remaining_merges / merge_rate if merge_rate > 0 else float("inf")
                eta_text = f"{eta_s / 60:.1f}m" if eta_s != float("inf") else "unknown"
                projected_finish_utc = (
                    (datetime.now(UTC) + timedelta(seconds=eta_s)).isoformat(timespec="seconds")
                    if eta_s != float("inf")
                    else "unknown"
                )
                emit_progress(
                    f"[train] merges={merges_completed}/{target_merges} "
                    f"vocab={len(self.vocabulary)}/{max_vocab_size} "
                    f"best_pair_count={highest_count} "
                    f"elapsed={elapsed_s / 60:.1f}m eta={eta_text} "
                    f"projected_finish_utc={projected_finish_utc} "
                    f"rate={merge_rate:.2f} merges/s"
                )
                last_progress_log_time = now
        total_elapsed_s = time.perf_counter() - start_time
        emit_progress(
            f"[train] done vocab={len(self.vocabulary)} elapsed={total_elapsed_s / 60:.1f}m"
        )


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    progress_every: int | None = None,
    progress_interval_seconds: float | None = None,
    include_timestamps: bool = False,
    progress_callback: Callable[[str], None] | None = None,
):
    bpe_tokenizer = BPE_Tokenizer(special_tokens)
    bpe_tokenizer.train(
        input_path,
        vocab_size,
        progress_every=progress_every,
        progress_interval_seconds=progress_interval_seconds,
        include_timestamps=include_timestamps,
        progress_callback=progress_callback,
    )
    vocab = bpe_tokenizer.vocabulary
    merges = bpe_tokenizer.merges
    return vocab, merges
