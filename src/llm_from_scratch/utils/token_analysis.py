"""Utilities for analyzing GPT-2 tokenization behavior.

Common usage:
    import tiktoken
    from utils import token_analysis as ta

    ta.tokenizer = tiktoken.get_encoding("gpt2")
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Any iterable[str] works; here we pass a single document.
    docs = [text]
    counts = ta.count_token_ids(docs)
    top_df = ta.top_n_token_df(counts, n=25)

Large-file streaming example:
    import tiktoken
    from utils import token_analysis as ta

    ta.tokenizer = tiktoken.get_encoding("gpt2")

    # Stream a large text file line-by-line to avoid reading it all at once.
    with open("data/very_large_corpus.txt", "r", encoding="utf-8") as f:
        counts = ta.count_token_ids(line for line in f)

    top_df = ta.top_n_token_df(counts, n=100)

Word-like unit example:
    import tiktoken
    from utils import token_analysis as ta

    ta.tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello world! This is a test."
    units = list(ta.wordlike_units(text))
    # Example output shape: ["Hello", "world!", "This", "is", "a", "test."]

Simple split word-frequency example:
    from utils import token_analysis as ta

    text = "Hello, world. Hello: test; world hello"
    top_df = ta.top_n_words_simple_split(text, n=3)
    # Example columns: ["word", "count", "fraction_total"]
"""

from collections import Counter
import re
from typing import Iterable, Iterator

import pandas as pd
import tiktoken

# Callers can set this directly, e.g.:
# tokenizer = tiktoken.get_encoding("gpt2")
tokenizer = None


def _get_tokenizer():
    global tokenizer
    if tokenizer is not None:
        return tokenizer
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception as exc:
        raise RuntimeError(
            "No tokenizer available. Set utils.token_analysis.tokenizer "
            "to your tiktoken GPT-2 encoding object."
        ) from exc
    return tokenizer


def encode(text: str) -> list[int]:
    """Encode text into GPT-2 token ids."""
    return _get_tokenizer().encode(text)


def count_token_ids(text_iter: Iterable[str]) -> Counter[int]:
    """Stream over text documents and count token-id frequencies."""
    counts: Counter[int] = Counter()
    for text in text_iter:
        counts.update(encode(text))
    return counts


def decode_id(tid: int) -> str:
    """Decode a single token id to its token string."""
    return _get_tokenizer().decode([tid])


def group_counts_by_lstrip(decoded_token_counts: Counter[str]) -> Counter[str]:
    """Group decoded-token counts by left-stripped token text.

    Example: " HER" and "HER" are combined under "HER".
    """
    grouped: Counter[str] = Counter()
    for token_text, count in decoded_token_counts.items():
        grouped[token_text.lstrip()] += count
    return grouped


def wordlike_units(text: str) -> Iterator[str]:
    """Yield word-like units using GPT token boundaries.

    A new unit starts when a decoded token begins with whitespace.
    Leading whitespace at a new unit boundary is stripped.
    """
    current: list[str] = []

    for tid in encode(text):
        token_text = decode_id(tid)

        if token_text and token_text[0].isspace():
            if current:
                yield "".join(current)
            current = [token_text.lstrip()]
        else:
            current.append(token_text)

    if current:
        yield "".join(current)


def top_n_token_df(token_id_counts: Counter[int], n: int) -> pd.DataFrame:
    """Return top-n token stats as a pandas DataFrame.

    Columns:
    - tokenid: token id
    - decoded_repr: repr(decoded token text)
    - count: raw count
    - fraction_total: count / total_count
    """
    total_count = sum(token_id_counts.values())
    if total_count == 0 or n <= 0:
        return pd.DataFrame(
            columns=["tokenid", "decoded_repr", "count", "fraction_total"]
        )

    rows = []
    for tid, count in token_id_counts.most_common(n):
        rows.append(
            {
                "tokenid": tid,
                "decoded_repr": repr(decode_id(tid)),
                "count": count,
                "fraction_total": count / total_count,
            }
        )

    return pd.DataFrame(rows)


def top_n_words_simple_split(text: str, n: int) -> pd.DataFrame:
    """Split text by whitespace/comma/colon/semicolon/period and return top-n words.

    Columns:
    - word: split unit
    - count: raw count
    - fraction_total: count / total_word_count
    """
    if n <= 0:
        return pd.DataFrame(columns=["word", "count", "fraction_total"])

    words = [w for w in re.split(r"[\s,.:;]+", text) if w]
    total_count = len(words)
    if total_count == 0:
        return pd.DataFrame(columns=["word", "count", "fraction_total"])

    counts = Counter(words)
    rows = []
    for word, count in counts.most_common(n):
        rows.append(
            {
                "word": word,
                "count": count,
                "fraction_total": count / total_count,
            }
        )

    return pd.DataFrame(rows)
