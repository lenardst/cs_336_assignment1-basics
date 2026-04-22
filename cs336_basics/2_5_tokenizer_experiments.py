import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# Minimal configuration.
SPECIAL_TOKEN = "<|endoftext|>"
DOC_DELIMITER = "<|endoftext|>"
N_SAMPLE_DOCS = 10
CHUNK_CHARS = 2_000_000
THROUGHPUT_MIN_BYTES = 50_000_000
PILE_BYTES = 825_000_000_000  # 825 GB

TINY_TRAIN = Path("data/TinyStoriesV2-GPT4-train.txt")
TINY_VALID = Path("data/TinyStoriesV2-GPT4-valid.txt")
OWT_TRAIN = Path("data/owt_train.txt")
OWT_VALID = Path("data/owt_valid.txt")

TINY_VOCAB = Path("data/tinystories_vocab_10000.pkl")
TINY_MERGES = Path("data/tinystories_merges_10000.pkl")
OWT_VOCAB = Path("data/owt_vocab_32000.pkl")
OWT_MERGES = Path("data/owt_merges_32000.pkl")

OUT_DIR = Path("data/tokenizer_experiments")
LOG_PATH = OUT_DIR / "metrics.log"
JSON_PATH = OUT_DIR / "metrics.json"


def iter_text_chunks(path: Path, chunk_chars: int = CHUNK_CHARS) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as file:
        while True:
            chunk = file.read(chunk_chars)
            if not chunk:
                break
            yield chunk


def iter_documents(path: Path, delimiter: str = DOC_DELIMITER) -> Iterator[str]:
    remainder = ""
    for chunk in iter_text_chunks(path):
        text = remainder + chunk
        parts = text.split(delimiter)
        for part in parts[:-1]:
            doc = part.strip()
            if doc:
                yield doc
        remainder = parts[-1]
    doc = remainder.strip()
    if doc:
        yield doc


def sample_first_n_documents(path: Path, n: int = N_SAMPLE_DOCS) -> list[str]:
    docs: list[str] = []
    for doc in iter_documents(path):
        docs.append(doc)
        if len(docs) >= n:
            break
    return docs


def load_tokenizer(vocab_path: Path, merges_path: Path) -> Any:
    tokenizer_module = importlib.import_module("cs336_basics.2_4_tokenizer")
    tokenizer_cls = getattr(tokenizer_module, "Tokenizer")
    return tokenizer_cls.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=[SPECIAL_TOKEN],
    )


def compression_ratio(tokenizer: Any, docs: list[str]) -> dict[str, float]:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    return {
        "total_bytes": float(total_bytes),
        "total_tokens": float(total_tokens),
        "bytes_per_token": float("inf") if total_tokens == 0 else total_bytes / total_tokens,
    }


def throughput_bytes_per_second(tokenizer: Any, input_path: Path) -> dict[str, float]:
    processed_bytes = 0
    processed_tokens = 0
    start = time.perf_counter()
    for doc in iter_documents(input_path):
        processed_bytes += len(doc.encode("utf-8"))
        processed_tokens += len(tokenizer.encode(doc))
        if processed_bytes >= THROUGHPUT_MIN_BYTES:
            break
    elapsed = time.perf_counter() - start
    bps = 0.0 if elapsed == 0 else processed_bytes / elapsed
    return {
        "benchmarked_bytes": float(processed_bytes),
        "benchmarked_tokens": float(processed_tokens),
        "elapsed_seconds": elapsed,
        "bytes_per_second": bps,
    }


def count_tokens(token_ids: Iterable[int]) -> tuple[int, int]:
    count = 0
    max_token_id = -1
    for token_id in token_ids:
        count += 1
        if token_id > max_token_id:
            max_token_id = token_id
    return count, max_token_id


def write_uint16_npy(token_ids: Iterable[int], output_path: Path, n_tokens: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(n_tokens,))
    i = 0
    for token_id in token_ids:
        arr[i] = token_id
        i += 1
    del arr


def encode_dataset(tokenizer: Any, input_path: Path, output_path: Path) -> dict[str, float | int | str]:
    start = time.perf_counter()
    ids_for_count = tokenizer.encode_iterable(iter_text_chunks(input_path))
    n_tokens, max_token_id = count_tokens(ids_for_count)
    if max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(f"Token id {max_token_id} does not fit in uint16")
    ids_for_write = tokenizer.encode_iterable(iter_text_chunks(input_path))
    write_uint16_npy(ids_for_write, output_path, n_tokens)
    elapsed = time.perf_counter() - start
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tokens": n_tokens,
        "max_token_id": max_token_id,
        "elapsed_seconds": elapsed,
    }


def ensure_paths_exist(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")


def main() -> None:
    if os.environ.get("TOKENIZER_EXPERIMENTS_WORKER") != "1":
        worker_env = os.environ.copy()
        worker_env["TOKENIZER_EXPERIMENTS_WORKER"] = "1"
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__))],
            env=worker_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print("Submitted tokenizer experiments asynchronously.")
        print(f"Worker PID: {worker.pid}")
        return

    ensure_paths_exist(
        [
            TINY_TRAIN,
            TINY_VALID,
            OWT_TRAIN,
            OWT_VALID,
            TINY_VOCAB,
            TINY_MERGES,
            OWT_VOCAB,
            OWT_MERGES,
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tiny_tokenizer = load_tokenizer(TINY_VOCAB, TINY_MERGES)
    owt_tokenizer = load_tokenizer(OWT_VOCAB, OWT_MERGES)

    tiny_docs = sample_first_n_documents(TINY_TRAIN)
    owt_docs = sample_first_n_documents(OWT_TRAIN)

    part_a_tiny = compression_ratio(tiny_tokenizer, tiny_docs)
    part_a_owt = compression_ratio(owt_tokenizer, owt_docs)
    part_b_cross = compression_ratio(tiny_tokenizer, owt_docs)

    tiny_tput = throughput_bytes_per_second(tiny_tokenizer, TINY_TRAIN)
    owt_tput = throughput_bytes_per_second(owt_tokenizer, OWT_TRAIN)

    pile_days_tiny = (PILE_BYTES / tiny_tput["bytes_per_second"]) / (3600 * 24)
    pile_days_owt = (PILE_BYTES / owt_tput["bytes_per_second"]) / (3600 * 24)

    enc_tiny_train = encode_dataset(tiny_tokenizer, TINY_TRAIN, OUT_DIR / "tiny_train_uint16.npy")
    enc_tiny_valid = encode_dataset(tiny_tokenizer, TINY_VALID, OUT_DIR / "tiny_valid_uint16.npy")
    enc_owt_train = encode_dataset(owt_tokenizer, OWT_TRAIN, OUT_DIR / "owt_train_uint16.npy")
    enc_owt_valid = encode_dataset(owt_tokenizer, OWT_VALID, OUT_DIR / "owt_valid_uint16.npy")

    results = {
        "part_a": {
            "tiny_tokenizer_on_tinystories_sample": part_a_tiny,
            "owt_tokenizer_on_owt_sample": part_a_owt,
        },
        "part_b": {
            "tiny_tokenizer_on_owt_sample": part_b_cross,
            "owt_tokenizer_on_owt_sample": part_a_owt,
            "delta_bytes_per_token_cross_minus_native": (
                part_b_cross["bytes_per_token"] - part_a_owt["bytes_per_token"]
            ),
        },
        "part_c": {
            "tiny_bytes_per_second": tiny_tput["bytes_per_second"],
            "owt_bytes_per_second": owt_tput["bytes_per_second"],
            "pile_days_tiny": pile_days_tiny,
            "pile_days_owt": pile_days_owt,
        },
        "part_d": {
            "encoded": {
                "tiny_train": enc_tiny_train,
                "tiny_valid": enc_tiny_valid,
                "owt_train": enc_owt_train,
                "owt_valid": enc_owt_valid,
            },
            "uint16_justification": (
                "uint16 is appropriate because both vocabularies (10k and 32k) "
                "are below 65,536 token IDs."
            ),
        },
    }

    with JSON_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2)

    lines = [
        f"(a) TinyStories tokenizer bytes/token on TinyStories sample: {part_a_tiny['bytes_per_token']:.6f}",
        f"(a) OpenWebText tokenizer bytes/token on OpenWebText sample: {part_a_owt['bytes_per_token']:.6f}",
        f"(b) TinyStories tokenizer bytes/token on OpenWebText sample: {part_b_cross['bytes_per_token']:.6f}",
        f"(b) Delta (cross - native): {results['part_b']['delta_bytes_per_token_cross_minus_native']:.6f}",
        f"(c) TinyStories tokenizer throughput (bytes/s): {tiny_tput['bytes_per_second']:.2f}",
        f"(c) OpenWebText tokenizer throughput (bytes/s): {owt_tput['bytes_per_second']:.2f}",
        f"(c) Estimated time for 825GB (TinyStories tokenizer): {pile_days_tiny:.2f} days",
        f"(c) Estimated time for 825GB (OpenWebText tokenizer): {pile_days_owt:.2f} days",
        "(d) uint16 is valid because max token ID is < 65536 for both vocabularies.",
        f"(d) Encoded arrays saved under: {OUT_DIR}",
    ]
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines) + "\n")

    print(f"Wrote {LOG_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
import argparse
import importlib
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

PILE_BYTES = 825_000_000_000  # 825 GB, decimal units.


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
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    for doc in iter_documents(path, delimiter, chunk_chars):
        seen += 1
        if len(sample) < n_docs:
            sample.append(doc)
            continue
        idx = rng.randint(1, seen)
        if idx <= n_docs:
            sample[idx - 1] = doc
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tokenizer experiments for CS336.")
    parser.add_argument("--tiny-train", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--tiny-valid", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--owt-train", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--owt-valid", type=Path, default=Path("data/owt_valid.txt"))

    parser.add_argument("--tiny-vocab", type=Path, default=Path("data/tinystories_vocab_10000.pkl"))
    parser.add_argument("--tiny-merges", type=Path, default=Path("data/tinystories_merges_10000.pkl"))
    parser.add_argument("--owt-vocab", type=Path, default=Path("data/owt_vocab_32000.pkl"))
    parser.add_argument("--owt-merges", type=Path, default=Path("data/owt_merges_32000.pkl"))

    parser.add_argument("--special-token", type=str, default="<|endoftext|>")
    parser.add_argument("--document-delimiter", type=str, default="<|endoftext|>")
    parser.add_argument("--n-sample-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-chars", type=int, default=2_000_000)
    parser.add_argument("--throughput-min-bytes", type=int, default=50_000_000)
    parser.add_argument("--uint16-buffer-tokens", type=int, default=2_000_000)

    parser.add_argument("--output-dir", type=Path, default=Path("data/tokenizer_experiments"))
    parser.add_argument("--log-file", type=Path, default=Path("data/tokenizer_experiments/metrics.log"))
    parser.add_argument("--skip-dataset-encoding", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    for required_path in [
        args.tiny_train,
        args.tiny_valid,
        args.owt_train,
        args.owt_valid,
        args.tiny_vocab,
        args.tiny_merges,
        args.owt_vocab,
        args.owt_merges,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing required path: {required_path}. "
                "Pass explicit paths via CLI if your files use different names."
            )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    special_tokens = [args.special_token]

    tiny_tokenizer = load_tokenizer(args.tiny_vocab, args.tiny_merges, special_tokens)
    owt_tokenizer = load_tokenizer(args.owt_vocab, args.owt_merges, special_tokens)

    tiny_docs, tiny_seen = reservoir_sample_documents(
        path=args.tiny_train,
        n_docs=args.n_sample_docs,
        delimiter=args.document_delimiter,
        chunk_chars=args.chunk_chars,
        seed=args.seed,
    )
    owt_docs, owt_seen = reservoir_sample_documents(
        path=args.owt_train,
        n_docs=args.n_sample_docs,
        delimiter=args.document_delimiter,
        chunk_chars=args.chunk_chars,
        seed=args.seed,
    )

    tiny_ratio = compression_ratio_bytes_per_token(tiny_tokenizer, tiny_docs)
    owt_ratio = compression_ratio_bytes_per_token(owt_tokenizer, owt_docs)
    cross_ratio = compression_ratio_bytes_per_token(tiny_tokenizer, owt_docs)

    tiny_tput = benchmark_throughput_bytes_per_second(
        tokenizer=tiny_tokenizer,
        path=args.tiny_train,
        delimiter=args.document_delimiter,
        min_bytes=args.throughput_min_bytes,
        chunk_chars=args.chunk_chars,
    )
    owt_tput = benchmark_throughput_bytes_per_second(
        tokenizer=owt_tokenizer,
        path=args.owt_train,
        delimiter=args.document_delimiter,
        min_bytes=args.throughput_min_bytes,
        chunk_chars=args.chunk_chars,
    )

    tiny_pile_seconds = (
        math.inf if tiny_tput["bytes_per_second"] == 0 else PILE_BYTES / tiny_tput["bytes_per_second"]
    )
    owt_pile_seconds = (
        math.inf if owt_tput["bytes_per_second"] == 0 else PILE_BYTES / owt_tput["bytes_per_second"]
    )

    dataset_encoding_results: dict[str, dict[str, float | int | str]] = {}
    if not args.skip_dataset_encoding:
        dataset_encoding_results["tiny_train"] = encode_dataset_file(
            tokenizer=tiny_tokenizer,
            input_path=args.tiny_train,
            output_path=output_dir / "tiny_train_uint16.npy",
            chunk_chars=args.chunk_chars,
        )
        dataset_encoding_results["tiny_valid"] = encode_dataset_file(
            tokenizer=tiny_tokenizer,
            input_path=args.tiny_valid,
            output_path=output_dir / "tiny_valid_uint16.npy",
            chunk_chars=args.chunk_chars,
        )
        dataset_encoding_results["owt_train"] = encode_dataset_file(
            tokenizer=owt_tokenizer,
            input_path=args.owt_train,
            output_path=output_dir / "owt_train_uint16.npy",
            chunk_chars=args.chunk_chars,
        )
        dataset_encoding_results["owt_valid"] = encode_dataset_file(
            tokenizer=owt_tokenizer,
            input_path=args.owt_valid,
            output_path=output_dir / "owt_valid_uint16.npy",
            chunk_chars=args.chunk_chars,
        )

    results: dict[str, Any] = {
        "config": {
            "tiny_train": str(args.tiny_train),
            "tiny_valid": str(args.tiny_valid),
            "owt_train": str(args.owt_train),
            "owt_valid": str(args.owt_valid),
            "tiny_vocab": str(args.tiny_vocab),
            "tiny_merges": str(args.tiny_merges),
            "owt_vocab": str(args.owt_vocab),
            "owt_merges": str(args.owt_merges),
            "special_token": args.special_token,
            "document_delimiter": args.document_delimiter,
            "n_sample_docs": args.n_sample_docs,
            "seed": args.seed,
            "chunk_chars": args.chunk_chars,
            "throughput_min_bytes": args.throughput_min_bytes,
            "skip_dataset_encoding": args.skip_dataset_encoding,
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

    metrics_json_path = output_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2, ensure_ascii=False)

    log_lines = [
        "Tokenizer experiment results",
        "",
        f"(a) Tiny tokenizer compression (bytes/token): {tiny_ratio['bytes_per_token']:.6f}",
        f"(a) OWT tokenizer compression (bytes/token): {owt_ratio['bytes_per_token']:.6f}",
        f"(b) Tiny tokenizer on OWT sample (bytes/token): {cross_ratio['bytes_per_token']:.6f}",
        (
            "(b) Delta vs OWT tokenizer on OWT sample "
            f"(cross - native): {results['part_b']['delta_bytes_per_token_cross_minus_native']:.6f}"
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
        log_lines.append("")
        log_lines.append("Dataset encoding outputs:")
        for key, value in dataset_encoding_results.items():
            log_lines.append(
                f"- {key}: {value['total_tokens']} tokens -> {value['output_path']} "
                f"in {value['elapsed_seconds']:.2f}s"
            )
    else:
        log_lines.append("")
        log_lines.append("Dataset encoding skipped via --skip-dataset-encoding.")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_lines) + "\n")

    print(f"Wrote sampled docs to: {samples_path}")
    print(f"Wrote metrics JSON to: {metrics_json_path}")
    print(f"Wrote metrics log to: {args.log_file}")
    if dataset_encoding_results:
        print("Wrote encoded datasets:")
        for key, value in dataset_encoding_results.items():
            print(f"  - {key}: {value['output_path']}")


if __name__ == "__main__":
    main()
