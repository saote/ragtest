# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run and split_text_on_tokens methods definition."""

from collections.abc import Iterable
from typing import Any

import tiktoken
from datashaper import ProgressTicker

import graphrag.config.defaults as defs
from graphrag.index.text_splitting import Tokenizer
from graphrag.index.verbs.text.chunk.typing import TextChunk


def run(
    input: list[str], args: dict[str, Any], tick: ProgressTicker
) -> Iterable[TextChunk]:
    """Chunks text into multiple parts. A pipeline verb."""
    tokens_per_chunk = args.get("chunk_size", defs.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)
    encoding_name = args.get("encoding_name", defs.ENCODING_MODEL)
    enc = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)

    return split_text_on_tokens(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
        tick,
    )


# Adapted from - https://github.com/langchain-ai/langchain/blob/77b359edf5df0d37ef0d539f678cf64f5557cb54/libs/langchain/langchain/text_splitter.py#L471
# So we could have better control over the chunking process
def split_text_on_tokens(
    texts: list[str], enc: Tokenizer, tick: ProgressTicker
) -> list[TextChunk]:
    """Split incoming text and return chunks."""
    result = []
    mapped_ids = []

    for source_doc_idx, text in enumerate(texts):
        encoded = enc.encode(text)
        tick(1)
        mapped_ids.append((source_doc_idx, encoded))

    input_ids: list[tuple[int, int]] = [
        (source_doc_idx, id) for source_doc_idx, ids in mapped_ids for id in ids
    ]

    start_idx = 0
    cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        chunk_text = enc.decode([id for _, id in chunk_ids])
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
        result.append(
            TextChunk(
                text_chunk=chunk_text,
                source_doc_indices=doc_indices,
                n_tokens=len(chunk_ids),
            )
        )
        start_idx += enc.tokens_per_chunk - enc.chunk_overlap
        cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result


### modified
def get_entity_info_multihop(text: str, entity_name: str)->str:
    """Get entity content in multi hop text."""
    entity_chunks = text.split('\n')
    entity_content = ''
    for lines in entity_chunks:
        if lines.startswith(entity_name + ':'):
            entity_content = lines[len(entity_name) + 2:]
    return entity_content


def split_text_on_tokens_multihop(
    texts: list[str], enc: Tokenizer, tick: ProgressTicker
) -> list[TextChunk]:
    """Split incoming text and return chunks."""
    result = []
    mapped_ids = []

    for source_doc_idx, text in enumerate(texts):
        body = get_entity_info_multihop(text, 'Body')
        title = get_entity_info_multihop(text, 'Title')
        source = get_entity_info_multihop(text, "Source")
        publish_time = get_entity_info_multihop(text, 'Published At')
        encoded = enc.encode(body)
        tick(1)
        mapped_ids.append((source_doc_idx, encoded, {'title':title, 'source': source, "publish_time": publish_time}))

    input_ids: list[tuple[int, int, dict]] = [
        (source_doc_idx, id, entity_dict) for source_doc_idx, ids, entity_dict in mapped_ids for id in ids
    ]

    start_idx = 0
    cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        chunk_text = enc.decode([id for _, id, _ in chunk_ids])
        doc_indices = list({doc_idx for doc_idx, _, _ in chunk_ids})
        entity_dict = chunk_ids[0][2]
        chunk_text += ('\nTitle: ' + entity_dict["title"])
        chunk_text += ('\nSource: ' + entity_dict["source"])
        chunk_text += ('\nPublish Time: ' + entity_dict["publish_time"])
        result.append(
            TextChunk(
                text_chunk=chunk_text,
                source_doc_indices=doc_indices,
                n_tokens=len(chunk_ids),
            )
        )
        start_idx += enc.tokens_per_chunk - enc.chunk_overlap
        cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result