import importlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import modal
import numpy as np

APP_NAME = "tokenizer-experiments"
DATA_DIR = Path("data")
TINY_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TINY_VALID = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
OWT_TRAIN = DATA_DIR / "owt_train.txt"
OWT_VALID = DATA_DIR / "owt_valid.txt"
TINY_VOCAB = DATA_DIR / "tinystories_vocab_10000.pkl"
TINY_MERGES = DATA_DIR / "tinystories_merges_10000.pkl"
OWT_VOCAB = DATA_DIR / "owt_vocab_32000.pkl"
OWT_MERGES = DATA_DIR / "owt_merges_32000.pkl"

SPECIAL_TOKEN = "<|endoftext|>"
DOC_DELIMITER = "<|endoftext|>"
PILE_BYTES = 825_000_000_000  # 825 GB, decimal units.

OUTPUT_VOLUME_NAME = "cs336-tokenizer-experiments"
REMOTE_WORKDIR = "/root/workspace"
REMOTE_DATA_DIR = f"{REMOTE_WORKDIR}/data"
REMOTE_OUTPUT_DIR = "/tokenizer_experiments_outputs"
ABC_LOG_FILENAME = "metrics_abc.log"
ABC_JSON_FILENAME = "metrics_abc.json"
PROGRESS_LOG_FILENAME = "progress.log"
DATASET_SHARD_TARGET_CHARS = 50_000_000

TOKENIZER_FILES = {
    "tiny": (TINY_VOCAB, TINY_MERGES),
    "owt": (OWT_VOCAB, OWT_MERGES),
}
ENCODING_DATASETS = {
    "tiny_train": (TINY_TRAIN, "tiny"),
    "tiny_valid": (TINY_VALID, "tiny"),
    "owt_train": (OWT_TRAIN, "owt"),
    "owt_valid": (OWT_VALID, "owt"),
}
_TOKENIZER_CACHE: dict[str, Any] = {}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(DATA_DIR), remote_path=REMOTE_DATA_DIR)
)
app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def iter_text_chunks(path: Path, chunk_chars: int) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as file:
        while True:
            chunk = file.read(chunk_chars)
            if not chunk:
                break
            yield chunk


def iter_documents(path: Path, delimiter: str, chunk_chars: int) -> Iterator[str]:
    remainder = ""
    for chunk in iter_text_chunks(path, chunk_chars):
        text = remainder + chunk
        parts = text.split(delimiter)
        for doc in parts[:-1]:
            cleaned = doc.strip()
            if cleaned:
                yield cleaned
        remainder = parts[-1]
    cleaned = remainder.strip()
    if cleaned:
        yield cleaned


def reservoir_sample_documents(
    path: Path,
    n_docs: int,
    delimiter: str,
    chunk_chars: int,
    seed: int,
) -> tuple[list[str], int]:
    sample: list[str] = []
    seen = 0
    for doc in iter_documents(path, delimiter, chunk_chars):
        seen += 1
        sample.append(doc)
        if len(sample) >= n_docs:
            break
    return sample, seen


def compression_ratio_bytes_per_token(tokenizer: Any, docs: list[str]) -> dict[str, float]:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    ratio = float("inf") if total_tokens == 0 else total_bytes / total_tokens
    return {
        "total_bytes": float(total_bytes),
        "total_tokens": float(total_tokens),
        "bytes_per_token": ratio,
    }


def benchmark_throughput_bytes_per_second(
    tokenizer: Any,
    path: Path,
    delimiter: str,
    min_bytes: int,
    chunk_chars: int,
) -> dict[str, float]:
    total_bytes = 0
    total_tokens = 0
    start = time.perf_counter()
    for doc in iter_documents(path, delimiter, chunk_chars):
        doc_bytes = len(doc.encode("utf-8"))
        total_bytes += doc_bytes
        total_tokens += len(tokenizer.encode(doc))
        if total_bytes >= min_bytes:
            break
    elapsed = time.perf_counter() - start
    throughput = 0.0 if elapsed == 0 else total_bytes / elapsed
    return {
        "benchmarked_bytes": float(total_bytes),
        "benchmarked_tokens": float(total_tokens),
        "elapsed_seconds": elapsed,
        "bytes_per_second": throughput,
    }


def encode_iterable_to_uint16_npy(
    token_ids: Iterable[int],
    output_path: Path,
    chunk_size: int = 2_000_000,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    chunk_lengths: list[int] = []
    total_tokens = 0
    max_token_id = -1
    current: list[int] = []

    with tempfile.TemporaryDirectory(prefix="token_chunks_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        chunk_idx = 0

        for token in token_ids:
            if token < 0 or token > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"Token ID {token} cannot fit into uint16. "
                    "Check vocab size or choose a larger dtype."
                )
            current.append(token)
            if token > max_token_id:
                max_token_id = token
            if len(current) >= chunk_size:
                arr = np.asarray(current, dtype=np.uint16)
                chunk_path = tmp_path / f"chunk_{chunk_idx:06d}.npy"
                np.save(chunk_path, arr)
                chunk_paths.append(chunk_path)
                chunk_lengths.append(arr.shape[0])
                total_tokens += arr.shape[0]
                chunk_idx += 1
                current.clear()

        if current:
            arr = np.asarray(current, dtype=np.uint16)
            chunk_path = tmp_path / f"chunk_{chunk_idx:06d}.npy"
            np.save(chunk_path, arr)
            chunk_paths.append(chunk_path)
            chunk_lengths.append(arr.shape[0])
            total_tokens += arr.shape[0]
            if arr.max(initial=0) > max_token_id:
                max_token_id = int(arr.max())

        final = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.uint16,
            shape=(total_tokens,),
        )
        cursor = 0
        for chunk_path, length in zip(chunk_paths, chunk_lengths):
            data = np.load(chunk_path, mmap_mode="r")
            final[cursor : cursor + length] = data
            cursor += length
        del final

    return {
        "total_tokens": total_tokens,
        "max_token_id": max_token_id,
        "dtype_max": int(np.iinfo(np.uint16).max),
    }


def encode_dataset_file(
    tokenizer: Any,
    input_path: Path,
    output_path: Path,
    chunk_chars: int,
) -> dict[str, float | int | str]:
    bytes_in = input_path.stat().st_size
    start = time.perf_counter()
    token_iter = tokenizer.encode_iterable(iter_text_chunks(input_path, chunk_chars))
    encoding_stats = encode_iterable_to_uint16_npy(token_iter, output_path)
    elapsed = time.perf_counter() - start
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_bytes": bytes_in,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (
            0.0 if elapsed == 0 else encoding_stats["total_tokens"] / elapsed
        ),
        "bytes_per_second": 0.0 if elapsed == 0 else bytes_in / elapsed,
        **encoding_stats,
    }


def load_tokenizer(vocab_path: Path, merges_path: Path, special_tokens: list[str]) -> Any:
    tokenizer_module = importlib.import_module("cs336_basics.2_4_tokenizer")
    tokenizer_cls = getattr(tokenizer_module, "Tokenizer")
    return tokenizer_cls.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=special_tokens,
    )


def _remote_data_path(local_path: Path) -> Path:
    return Path(REMOTE_DATA_DIR) / local_path.name


def _get_tokenizer_cached(tokenizer_key: str) -> Any:
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_key)
    if tokenizer is not None:
        return tokenizer
    vocab_local_path, merges_local_path = TOKENIZER_FILES[tokenizer_key]
    tokenizer = load_tokenizer(
        _remote_data_path(vocab_local_path),
        _remote_data_path(merges_local_path),
        [SPECIAL_TOKEN],
    )
    _TOKENIZER_CACHE[tokenizer_key] = tokenizer
    return tokenizer


def _build_log_lines(
    results: dict[str, Any],
    dataset_encoding_results: dict[str, dict[str, float | int | str]],
) -> list[str]:
    tiny_ratio = results["part_a"]["tiny_tokenizer_on_tiny_sample"]
    owt_ratio = results["part_a"]["owt_tokenizer_on_owt_sample"]
    cross_ratio = results["part_b"]["tiny_tokenizer_on_owt_sample"]
    tiny_tput = results["part_c"]["tiny_tokenizer_throughput"]
    owt_tput = results["part_c"]["owt_tokenizer_throughput"]
    lines = [
        "Tokenizer experiment results",
        "",
        f"(a) Tiny tokenizer compression (bytes/token): {tiny_ratio['bytes_per_token']:.6f}",
        f"(a) OWT tokenizer compression (bytes/token): {owt_ratio['bytes_per_token']:.6f}",
        f"(b) Tiny tokenizer on OWT sample (bytes/token): {cross_ratio['bytes_per_token']:.6f}",
        (
            "(b) Delta vs OWT tokenizer on OWT sample "
            f"(cross - native): {cross_ratio['bytes_per_token'] - owt_ratio['bytes_per_token']:.6f}"
        ),
        f"(c) Tiny throughput (bytes/s): {tiny_tput['bytes_per_second']:.2f}",
        f"(c) OWT throughput (bytes/s): {owt_tput['bytes_per_second']:.2f}",
        (
            "(c) Estimated Pile tokenization time with tiny tokenizer: "
            f"{results['part_c']['pile_estimate_days_tiny']:.2f} days"
        ),
        (
            "(c) Estimated Pile tokenization time with OWT tokenizer: "
            f"{results['part_c']['pile_estimate_days_owt']:.2f} days"
        ),
        "(d) uint16 rationale: vocab sizes 10k and 32k fit in [0, 65535].",
    ]
    if dataset_encoding_results:
        lines.append("")
        lines.append("Dataset encoding outputs:")
        for key, value in dataset_encoding_results.items():
            lines.append(
                f"- {key}: {value['total_tokens']} tokens -> {value['output_path']} "
                f"in {value['elapsed_seconds']:.2f}s"
            )
    else:
        lines.append("")
        lines.append("Dataset encoding pending or skipped.")
    return lines


@app.function(image=image, volumes={REMOTE_OUTPUT_DIR: output_volume}, timeout=24 * 60 * 60)
def encode_dataset_shard_remote(
    tokenizer_key: str,
    shard_input_path: str,
    shard_output_path: str,
    chunk_chars: int = 2_000_000,
) -> dict[str, float | int | str]:
    os.chdir(REMOTE_WORKDIR)
    tokenizer = _get_tokenizer_cached(tokenizer_key)
    result = encode_dataset_file(
        tokenizer=tokenizer,
        input_path=Path(shard_input_path),
        output_path=Path(shard_output_path),
        chunk_chars=chunk_chars,
    )
    output_volume.commit()
    return result


def _split_file_into_delimiter_shards(
    input_path: Path,
    shard_dir: Path,
    delimiter: str,
    target_chars: int,
) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    buffer = ""
    delimiter_len = len(delimiter)
    read_size = max(1_000_000, target_chars // 4)

    def flush_shard(text: str, shard_idx: int) -> Path:
        shard_path = shard_dir / f"shard_{shard_idx:06d}.txt"
        shard_path.write_text(text, encoding="utf-8")
        return shard_path

    with input_path.open("r", encoding="utf-8") as input_file:
        while True:
            chunk = input_file.read(read_size)
            if not chunk:
                break
            buffer += chunk
            while len(buffer) >= target_chars:
                split_idx = buffer.rfind(delimiter, 0, target_chars)
                if split_idx == -1:
                    split_idx = buffer.find(delimiter, target_chars)
                if split_idx == -1:
                    split_idx = target_chars
                else:
                    split_idx += delimiter_len
                shard_text = buffer[:split_idx]
                if shard_text:
                    shard_paths.append(flush_shard(shard_text, len(shard_paths)))
                buffer = buffer[split_idx:]

    if buffer:
        shard_paths.append(flush_shard(buffer, len(shard_paths)))
    if not shard_paths:
        shard_paths.append(flush_shard("", 0))
    return shard_paths


def _merge_uint16_shards(shard_npy_paths: list[Path], output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_lengths: list[int] = []
    total_tokens = 0
    max_token_id = -1

    for shard_npy_path in shard_npy_paths:
        shard_arr = np.load(shard_npy_path, mmap_mode="r")
        shard_len = int(shard_arr.shape[0])
        shard_lengths.append(shard_len)
        total_tokens += shard_len
        if shard_len > 0:
            shard_max = int(shard_arr.max())
            if shard_max > max_token_id:
                max_token_id = shard_max

    merged = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))
    cursor = 0
    for shard_npy_path, shard_len in zip(shard_npy_paths, shard_lengths):
        shard_arr = np.load(shard_npy_path, mmap_mode="r")
        merged[cursor : cursor + shard_len] = shard_arr
        cursor += shard_len
    del merged
    return total_tokens, max_token_id


@app.function(image=image, volumes={REMOTE_OUTPUT_DIR: output_volume}, timeout=24 * 60 * 60)
def encode_one_dataset_remote(
    dataset_key: str,
    chunk_chars: int = 2_000_000,
    shard_target_chars: int = DATASET_SHARD_TARGET_CHARS,
) -> dict[str, float | int | str]:
    if dataset_key not in ENCODING_DATASETS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    os.chdir(REMOTE_WORKDIR)
    local_input_path, tokenizer_key = ENCODING_DATASETS[dataset_key]
    input_path = _remote_data_path(local_input_path)
    start = time.perf_counter()
    shard_dir = Path(REMOTE_OUTPUT_DIR) / f"{dataset_key}_shards"
    shard_text_paths = _split_file_into_delimiter_shards(
        input_path=input_path,
        shard_dir=shard_dir,
        delimiter=DOC_DELIMITER,
        target_chars=shard_target_chars,
    )
    output_volume.commit()
    shard_fn = modal.Function.from_name(APP_NAME, "encode_dataset_shard_remote")
    shard_calls: list[tuple[int, Path, Any]] = []
    for shard_idx, shard_text_path in enumerate(shard_text_paths):
        shard_npy_path = shard_dir / f"encoded_{shard_idx:06d}.npy"
        shard_call = shard_fn.spawn(
            tokenizer_key=tokenizer_key,
            shard_input_path=str(shard_text_path),
            shard_output_path=str(shard_npy_path),
            chunk_chars=chunk_chars,
        )
        shard_calls.append((shard_idx, shard_npy_path, shard_call))

    shard_calls.sort(key=lambda item: item[0])
    shard_results: list[dict[str, float | int | str]] = []
    shard_npy_paths: list[Path] = []
    for _, shard_npy_path, shard_call in shard_calls:
        shard_results.append(shard_call.get())
        shard_npy_paths.append(shard_npy_path)

    output_volume.reload()
    output_path = Path(REMOTE_OUTPUT_DIR) / f"{dataset_key}_uint16.npy"
    total_tokens, max_token_id = _merge_uint16_shards(shard_npy_paths, output_path)
    elapsed = time.perf_counter() - start
    bytes_in = input_path.stat().st_size
    result: dict[str, float | int | str] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_bytes": bytes_in,
        "elapsed_seconds": elapsed,
        "tokens_per_second": 0.0 if elapsed == 0 else total_tokens / elapsed,
        "bytes_per_second": 0.0 if elapsed == 0 else bytes_in / elapsed,
        "total_tokens": total_tokens,
        "max_token_id": max_token_id,
        "dtype_max": int(np.iinfo(np.uint16).max),
        "shard_count": len(shard_text_paths),
    }
    result["dataset_key"] = dataset_key
    result["shards"] = shard_results
    status_path = Path(REMOTE_OUTPUT_DIR) / f"{dataset_key}_encoding.json"
    with status_path.open("w", encoding="utf-8") as status_file:
        json.dump(result, status_file, indent=2)
    output_volume.commit()
    return result


@app.function(image=image, volumes={REMOTE_OUTPUT_DIR: output_volume}, timeout=24 * 60 * 60)
def run_tokenizer_experiments_remote(
    n_sample_docs: int = 10,
    seed: int = 42,
    chunk_chars: int = 2_000_000,
    throughput_min_bytes: int = 5_000_000,
    skip_dataset_encoding: bool = False,
) -> str:
    os.chdir(REMOTE_WORKDIR)
    tiny_train = _remote_data_path(TINY_TRAIN)
    tiny_valid = _remote_data_path(TINY_VALID)
    owt_train = _remote_data_path(OWT_TRAIN)
    owt_valid = _remote_data_path(OWT_VALID)
    tiny_vocab = _remote_data_path(TINY_VOCAB)
    tiny_merges = _remote_data_path(TINY_MERGES)
    owt_vocab = _remote_data_path(OWT_VOCAB)
    owt_merges = _remote_data_path(OWT_MERGES)
    output_dir = Path(REMOTE_OUTPUT_DIR)
    progress_log_path = output_dir / PROGRESS_LOG_FILENAME

    def log_progress(message: str, commit: bool = False) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line)
        with progress_log_path.open("a", encoding="utf-8") as progress_file:
            progress_file.write(line + "\n")
        if commit:
            output_volume.commit()

    for required_path in [
        tiny_train,
        tiny_valid,
        owt_train,
        owt_valid,
        tiny_vocab,
        tiny_merges,
        owt_vocab,
        owt_merges,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing required path: {required_path}. "
                "Ensure the file exists in the local data/ directory before running Modal."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")
    log_progress("Started run_tokenizer_experiments_remote", commit=True)
    special_tokens = [SPECIAL_TOKEN]

    log_progress("Loading tokenizers")
    tiny_tokenizer = load_tokenizer(tiny_vocab, tiny_merges, special_tokens)
    owt_tokenizer = load_tokenizer(owt_vocab, owt_merges, special_tokens)
    log_progress("Loaded tokenizers")

    log_progress("Sampling TinyStories documents")
    tiny_docs, tiny_seen = reservoir_sample_documents(
        path=tiny_train,
        n_docs=n_sample_docs,
        delimiter=DOC_DELIMITER,
        chunk_chars=chunk_chars,
        seed=seed,
    )
    log_progress(f"Sampled TinyStories documents (seen={tiny_seen})")
    log_progress("Sampling OpenWebText documents")
    owt_docs, owt_seen = reservoir_sample_documents(
        path=owt_train,
        n_docs=n_sample_docs,
        delimiter=DOC_DELIMITER,
        chunk_chars=chunk_chars,
        seed=seed,
    )
    log_progress(f"Sampled OpenWebText documents (seen={owt_seen})")

    tiny_ratio = compression_ratio_bytes_per_token(tiny_tokenizer, tiny_docs)
    owt_ratio = compression_ratio_bytes_per_token(owt_tokenizer, owt_docs)
    cross_ratio = compression_ratio_bytes_per_token(tiny_tokenizer, owt_docs)

    log_progress("Benchmarking TinyStories throughput")
    tiny_tput = benchmark_throughput_bytes_per_second(
        tokenizer=tiny_tokenizer,
        path=tiny_train,
        delimiter=DOC_DELIMITER,
        min_bytes=throughput_min_bytes,
        chunk_chars=chunk_chars,
    )
    log_progress("Benchmarking OpenWebText throughput")
    owt_tput = benchmark_throughput_bytes_per_second(
        tokenizer=owt_tokenizer,
        path=owt_train,
        delimiter=DOC_DELIMITER,
        min_bytes=throughput_min_bytes,
        chunk_chars=chunk_chars,
    )
    log_progress("Finished part a-c metrics")

    tiny_pile_seconds = (
        math.inf if tiny_tput["bytes_per_second"] == 0 else PILE_BYTES / tiny_tput["bytes_per_second"]
    )
    owt_pile_seconds = (
        math.inf if owt_tput["bytes_per_second"] == 0 else PILE_BYTES / owt_tput["bytes_per_second"]
    )

    dataset_encoding_results: dict[str, dict[str, float | int | str]] = {}
    results: dict[str, Any] = {
        "config": {
            "tiny_train": str(tiny_train),
            "tiny_valid": str(tiny_valid),
            "owt_train": str(owt_train),
            "owt_valid": str(owt_valid),
            "tiny_vocab": str(tiny_vocab),
            "tiny_merges": str(tiny_merges),
            "owt_vocab": str(owt_vocab),
            "owt_merges": str(owt_merges),
            "special_token": SPECIAL_TOKEN,
            "document_delimiter": DOC_DELIMITER,
            "n_sample_docs": n_sample_docs,
            "seed": seed,
            "chunk_chars": chunk_chars,
            "throughput_min_bytes": throughput_min_bytes,
            "skip_dataset_encoding": skip_dataset_encoding,
        },
        "sampling": {
            "tiny_documents_seen": tiny_seen,
            "owt_documents_seen": owt_seen,
            "tiny_sampled_docs_count": len(tiny_docs),
            "owt_sampled_docs_count": len(owt_docs),
        },
        "part_a": {
            "tiny_tokenizer_on_tiny_sample": tiny_ratio,
            "owt_tokenizer_on_owt_sample": owt_ratio,
        },
        "part_b": {
            "tiny_tokenizer_on_owt_sample": cross_ratio,
            "owt_tokenizer_on_owt_sample": owt_ratio,
            "delta_bytes_per_token_cross_minus_native": (
                cross_ratio["bytes_per_token"] - owt_ratio["bytes_per_token"]
            ),
        },
        "part_c": {
            "tiny_tokenizer_throughput": tiny_tput,
            "owt_tokenizer_throughput": owt_tput,
            "pile_estimate_seconds_tiny": tiny_pile_seconds,
            "pile_estimate_hours_tiny": tiny_pile_seconds / 3600,
            "pile_estimate_days_tiny": tiny_pile_seconds / (3600 * 24),
            "pile_estimate_seconds_owt": owt_pile_seconds,
            "pile_estimate_hours_owt": owt_pile_seconds / 3600,
            "pile_estimate_days_owt": owt_pile_seconds / (3600 * 24),
        },
        "part_d": {
            "encoding_results": dataset_encoding_results,
            "uint16_justification": (
                "uint16 is appropriate because both vocabularies are < 65,536 tokens "
                "(10k and 32k), so every token ID fits exactly while using half the "
                "memory of int32."
            ),
        },
    }

    abc_results = {
        "config": results["config"],
        "sampling": results["sampling"],
        "part_a": results["part_a"],
        "part_b": results["part_b"],
        "part_c": results["part_c"],
        "part_d": {
            "status": (
                "skipped" if skip_dataset_encoding else "running in parallel; check per-dataset status files"
            ),
            "uint16_justification": results["part_d"]["uint16_justification"],
        },
    }

    abc_json_path = output_dir / ABC_JSON_FILENAME
    with abc_json_path.open("w", encoding="utf-8") as abc_json_file:
        json.dump(abc_results, abc_json_file, indent=2, ensure_ascii=False)

    abc_log_path = output_dir / ABC_LOG_FILENAME
    with abc_log_path.open("w", encoding="utf-8") as abc_log_file:
        abc_log_file.write(
            "\n".join(
                _build_log_lines(
                    results=results,
                    dataset_encoding_results={},
                )
            )
            + "\n"
        )
    output_volume.commit()
    log_progress("Committed early part a-c metrics", commit=True)

    metrics_json_path = output_dir / "metrics.json"
    log_path = output_dir / "metrics.log"

    if not skip_dataset_encoding:
        encode_fn = modal.Function.from_name(APP_NAME, "encode_one_dataset_remote")
        for dataset_key in ENCODING_DATASETS:
            log_progress(f"Starting dataset encoding: {dataset_key}", commit=True)
            call = encode_fn.spawn(dataset_key=dataset_key, chunk_chars=chunk_chars)
            dataset_encoding_results[dataset_key] = call.get()  # type: ignore[assignment]
            results["part_d"]["encoding_results"] = dataset_encoding_results
            with metrics_json_path.open("w", encoding="utf-8") as metrics_file:
                json.dump(results, metrics_file, indent=2, ensure_ascii=False)
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("\n".join(_build_log_lines(results=results, dataset_encoding_results=dataset_encoding_results)) + "\n")
            output_volume.commit()
            log_progress(f"Finished dataset encoding: {dataset_key}", commit=True)

    results["part_d"]["encoding_results"] = dataset_encoding_results

    samples_path = output_dir / "sampled_documents.json"
    with samples_path.open("w", encoding="utf-8") as sample_file:
        json.dump(
            {
                "tiny_documents": tiny_docs,
                "owt_documents": owt_docs,
            },
            sample_file,
            indent=2,
            ensure_ascii=False,
        )

    with metrics_json_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2, ensure_ascii=False)

    log_lines = _build_log_lines(results=results, dataset_encoding_results=dataset_encoding_results)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_lines) + "\n")

    output_volume.commit()
    log_progress("Completed run_tokenizer_experiments_remote", commit=True)
    print(f"Wrote early metrics to: {abc_log_path} and {abc_json_path}")
    print(f"Wrote sampled docs to: {samples_path}")
    print(f"Wrote metrics JSON to: {metrics_json_path}")
    print(f"Wrote metrics log to: {log_path}")
    if dataset_encoding_results:
        print("Wrote encoded datasets:")
        for key, value in dataset_encoding_results.items():
            print(f"  - {key}: {value['output_path']}")
    return "\n".join(log_lines)


@app.local_entrypoint()
def main(
    n_sample_docs: int = 10,
    seed: int = 42,
    chunk_chars: int = 2_000_000,
    throughput_min_bytes: int = 5_000_000,
    skip_dataset_encoding: bool = False,
) -> None:
    try:
        deployed_fn = modal.Function.from_name(APP_NAME, "run_tokenizer_experiments_remote")
    except Exception as exc:
        raise RuntimeError(
            "Could not find deployed Modal function 'run_tokenizer_experiments_remote'. "
            f"Deploy first with: modal deploy {Path(__file__).as_posix()}"
        ) from exc

    function_call = deployed_fn.spawn(
        n_sample_docs=n_sample_docs,
        seed=seed,
        chunk_chars=chunk_chars,
        throughput_min_bytes=throughput_min_bytes,
        skip_dataset_encoding=skip_dataset_encoding,
    )
    print("Submitted tokenizer experiments job asynchronously.")
    print(f"FunctionCall ID: {function_call.object_id}")
    print(
        f"Artifacts will be written to Modal Volume '{OUTPUT_VOLUME_NAME}' at "
        f"{REMOTE_OUTPUT_DIR}: {ABC_LOG_FILENAME}, {ABC_JSON_FILENAME}, metrics.log, metrics.json, "
        "sampled_documents.json, and encoded .npy files."
    )


if __name__ == "__main__":
    main()
