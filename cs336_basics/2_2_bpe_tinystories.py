from bpe_tokenizer import train_bpe
import pickle
import os
import time

input_path = "data/TinyStoriesV2-GPT4-valid.txt"
max_vocab_size = 10_000
special_tokens = ["<|endoftext|>"]

start_time = time.perf_counter()
vocab, merges = train_bpe(input_path, max_vocab_size, special_tokens)
elapsed_s = time.perf_counter() - start_time

os.makedirs("data", exist_ok=True)
vocab_path = "data/tinystories_vocab_10000.pkl"
merges_path = "data/tinystories_merges_10000.pkl"
with open(vocab_path, "wb") as vocab_file:
    pickle.dump(vocab, vocab_file)
with open(merges_path, "wb") as merges_file:
    pickle.dump(merges, merges_file)

longest_token = max(vocab.values(), key=len)
print(f"Saved vocab to: {vocab_path}")
print(f"Saved merges to: {merges_path}")
print(f"Vocabulary size: {len(vocab)}")
print(f"Training time: {elapsed_s:.2f}s")
print(f"Longest token length (bytes): {len(longest_token)}")
print(f"Longest token (utf-8): {longest_token.decode('utf-8', errors='replace')!r}")

# 459.87 MB, 89.48s, longest token: ' accomplishment'

# The py-spy flamegraph shows that most time is spent in the BPE merge loop, with update_word_pairs (32.38%) and merge_tokens_words_element (32.16%) together accounting for about two-thirds of sampled execution time. Smaller portions go to loop overhead and preprocessing/counting, and because subprocess profiling was not enabled, worker-process pretokenization time may be underrepresented.
