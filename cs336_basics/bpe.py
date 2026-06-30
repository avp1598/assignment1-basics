from multiprocessing import Pool
import os
from typing import BinaryIO
import regex
from typing import Iterator
from regex import Match
import json
import time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(corpus: str) -> Iterator[Match[str]]:
    """
    Pretokenize a corpus.
    """
    PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'"
    return regex.finditer(PAT, corpus)


# def find_most_frequent_pair(token_state: dict[tuple[bytes, ...], int], pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, ...]:
#     pairs: dict[tuple[bytes, ...], int] = {}
#     for word, count in token_state.items():
#         for i in range(len(word) - 1):
#             pair = word[i:i+2]
#             if pair in pairs:
#                 pairs[pair] += count
#             else:
#                 pairs[pair] = count
#     max_count = max(pairs.values(), default=0)
#     max_pairs = [pair for pair, count in pairs.items() if count == max_count]
#     chosen_pair = max(max_pairs)
#     return chosen_pair

def initialize_pair_counts(token_state: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], int]:
    pairs: dict[tuple[bytes, ...], int] = {}
    for word, count in token_state.items():
        for i in range(len(word) - 1):
            pair = word[i:i+2]
            if pair in pairs:
                pairs[pair] += count
            else:
                pairs[pair] = count
    return pairs

def find_most_frequent_pair(pair_counts: dict[tuple[bytes, ...], int]) -> tuple[bytes, ...]:
    max_count = max(pair_counts.values(), default=0)
    max_pairs = [pair for pair, count in pair_counts.items() if count == max_count]
    chosen_pair = max(max_pairs)
    return chosen_pair
        
def merge_token_state(token_state: dict[tuple[bytes, ...], int], chosen_pair: tuple[bytes, ...], pair_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], int]:
    new_token_state: dict[tuple[bytes, ...], int] = {}
    for word, count in token_state.items():
        merged_word: list[bytes] = []
        i = 0
        while i < len(word):
            # Check if the next bigram matches the chosen_pair
            if (
                i < len(word) - 1
                and word[i] == chosen_pair[0]
                and word[i + 1] == chosen_pair[1]
            ):
                merged_word.append(word[i] + word[i + 1])
                i += 2  # Skip next component since merged
            else:
                merged_word.append(word[i])
                i += 1

        merged_tuple = tuple(merged_word)
        if merged_tuple != word:
            for j in range(len(word) - 1):
                pair = word[j:j + 2]
                pair_counts[pair] -= count
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            for j in range(len(merged_word) - 1):
                pair = (merged_word[j], merged_word[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count

        if merged_tuple in new_token_state:
            new_token_state[merged_tuple] += count
        else:
            new_token_state[merged_tuple] = count

    
   
    return new_token_state

def pretokenize_chunk(input_path: str | str, start: int, end: int, special_tokens: list[str]) -> dict[str, int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        counts: dict[str, int] = {}

        split_chunks = regex.split("|".join([regex.escape(tok) for tok in special_tokens]), chunk.decode("utf-8", errors="ignore"))
   
        for split_chunk in split_chunks:
            for token in pretokenize(split_chunk):
                t = token.group(0)
                counts[t] = counts.get(t, 0) + 1
        return counts

def train_bpe(input_path: str | str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on a corpus.
    """
    vocab: dict[int, bytes] = {}

    # First, add special tokens to the vocab
    for idx, token in enumerate[str](special_tokens):
        vocab[idx] = token.encode("utf-8")

    # Next, add all 256 single-byte values
    next_idx = len(special_tokens)
    for b in range(256):
        vocab[next_idx + b] = bytes([b])

    merges: list[tuple[bytes, bytes]] = []
    word_counts: dict[str, int] = {}

    boundaries: list[int] = []

    start_time = time.time()
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count() or 4
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
    end_time = time.time()
    print(f"Time taken to find chunk boundaries: {end_time - start_time} seconds")

    start_time = time.time()
    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    end_time = time.time()
    print(f"Time taken to create chunk arguments: {end_time - start_time} seconds")

    start_time = time.time()
    with Pool(processes=num_processes) as p:
        partial_dicts = p.starmap(pretokenize_chunk, chunk_args)
    end_time = time.time()
    print(f"Time taken to pretokenize chunks: {end_time - start_time} seconds")

    start_time = time.time()
    for partial in partial_dicts:
        for token, count in partial.items():
            word_counts[token] = word_counts.get(token, 0) + count
    end_time = time.time()
    print(f"Time taken to count words: {end_time - start_time} seconds")

    start_time = time.time()
    token_state: dict[tuple[bytes, ...], int] = {}
    for token, count in word_counts.items():
        encoded = token.encode("utf-8")
        token_state[tuple(bytes([b]) for b in encoded)] = count
    end_time = time.time()
    print(f"Time taken to create token state: {end_time - start_time} seconds")

    pair_counts: dict[tuple[bytes, ...], int] = initialize_pair_counts(token_state)

    start_time = time.time()
    while len(vocab) < vocab_size:
        if not pair_counts: break

        chosen_pair = find_most_frequent_pair(pair_counts)
        merges.append((chosen_pair[0], chosen_pair[1]))
        token_state = merge_token_state(token_state, chosen_pair, pair_counts)
        vocab[len(vocab)] = chosen_pair[0] + chosen_pair[1]
        
    end_time = time.time()
    print(f"Time taken to merge tokens: {end_time - start_time} seconds")

    start_time = time.time()
    # serialize the vocab and merges
    # Convert bytes to str for JSON serialization
    vocab_serializable = {k: v.decode('utf-8', errors='replace') for k, v in vocab.items()}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=4)
    with open("merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{merge[0].decode('utf-8', errors='replace')} {merge[1].decode('utf-8', errors='replace')}\n")
    end_time = time.time()
    print(f"Time taken to serialize vocab and merges: {end_time - start_time} seconds")

    return vocab, merges


if __name__ == "__main__":
    print(train_bpe(input_path="data/test.txt", vocab_size=2603, special_tokens=["<|endoftext|>"]))