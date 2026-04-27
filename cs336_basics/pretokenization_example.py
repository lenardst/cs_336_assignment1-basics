import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    step = 4096
    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            block = file.read(step)
            if not block:
                boundaries[bi] = file_size
                break
            i = block.find(split_special_token)
            if i != -1:
                boundaries[bi] = pos + i
                break
            pos += step
    return sorted(set(boundaries))


# with open(..., "rb") as f:
#     b = find_chunk_boundaries(f, 4, b"<|endoftext|>")
#     for s, e in zip(b[:-1], b[1:]):
#         f.seek(s)
#         chunk = f.read(e - s).decode("utf-8", errors="ignore")
