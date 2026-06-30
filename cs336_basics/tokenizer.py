import json
from typing import Iterator, Iterable
from regex import Match
import regex


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens: list[str] = special_tokens if special_tokens else []

        self.merge_rank = {merge: i for i, merge in enumerate(merges)}
        self.bytes_to_id = {v: k for k, v in vocab.items()}
 

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
            vocab = {int(k): v.encode("utf-8") for k, v in raw.items()}
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges: list[tuple[bytes, bytes]] = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(regex.escape(tok) for tok in sorted_special_tokens)
            split_chunks = regex.split(f"({pattern})", text)
        else:
            split_chunks = [text]

        for split_chunk in split_chunks:
            if split_chunk == "":
                continue

            if self.special_tokens and split_chunk in self.special_tokens:
                ids.append(self.bytes_to_id[split_chunk.encode("utf-8")])
                continue

            pretokenized = self._pretokenize(split_chunk)
            for match in pretokenized:
                token = match.group(0)
                token_tuple: tuple[bytes, ...] = tuple(bytes([b]) for b in token.encode("utf-8"))

                pieces = list(token_tuple)
                while len(pieces) > 1:
                    best_rank = float('inf')
                    best_i = None
                    # Find best pair to merge
                    for i in range(len(pieces) - 1):
                        pair = (pieces[i], pieces[i+1])
                        if pair in self.merge_rank and self.merge_rank[pair] < best_rank:
                            best_rank = self.merge_rank[pair]
                            best_i = i
                    if best_i is None:
                        break  # done with this pretoken
                    # Merge at best_i
                    merged = pieces[best_i] + pieces[best_i + 1]
                    pieces = pieces[:best_i] + [merged] + pieces[best_i+2:]

                for piece in pieces:
                    ids.append(self.bytes_to_id[piece])

        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable:
            yield from self.encode(item)
            
    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def _pretokenize(self, corpus: str) -> Iterator[Match[str]]:
        """
        Pretokenize a corpus.
        """
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"
        return regex.finditer(PAT, corpus)


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files("vocab.json", "merges.txt", ["<|endoftext|>"])
    print(tokenizer.encode("Hello"))