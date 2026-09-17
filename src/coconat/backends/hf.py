"""Hugging Face BIO token classifier with complete-word packing and safe offsets."""

from __future__ import annotations

import inspect

import numpy as np

from .base import Backend
from ..schema import Evidence


def chunk_word_pieces(word_pieces, budget):
    """Return whole-word chunks; fail rather than silently truncate a giant word."""
    chunks, current, size = [], [], 0
    for word_index, pieces in enumerate(word_pieces):
        if not pieces or len(pieces) > budget:
            raise ValueError(f"Word {word_index} cannot fit the model input budget")
        if current and size + len(pieces) > budget:
            chunks.append(current)
            current, size = [], 0
        current.append((word_index, pieces))
        size += len(pieces)
    if current:
        chunks.append(current)
    return chunks


class HFBackend(Backend):
    embedding_description = "mean pooled final hidden states of the NER backbone"

    def __init__(self, checkpoint, labels, device="auto", batch_size=8,
                 max_length=512, revision=None, tokenizer=None, **kwargs):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError('Install the encoder dependencies: pip install -e ".[hf]"') from exc
        self.torch = torch
        self.labels = tuple(labels)
        self.name = checkpoint
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or checkpoint, revision=revision,
                                                       use_fast=True, trust_remote_code=False)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer with word_ids() is required")
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint, revision=revision,
                                                                    trust_remote_code=False)
        self.model.to(self.device).eval()
        self.tags = tuple(self.model.config.id2label[i] for i in range(self.model.config.num_labels))
        expected = {"O"} | {tag for label in labels for tag in ("B-" + label, "I-" + label)}
        if set(self.tags) != expected:
            raise ValueError(f"Checkpoint must have the dataset BIO head. Expected {sorted(expected)}, "
                             f"found {self.tags}. Fine-tune the checkpoint first; do not relabel LABEL_0 blindly.")
        tokenizer_limit = self.tokenizer.model_max_length
        self.max_length = min(int(max_length), int(tokenizer_limit))
        # RoBERTa-family position tables include a padding offset.
        model_limit = getattr(self.model.config, "max_position_embeddings", self.max_length)
        if getattr(self.model.config, "model_type", "") in {"roberta", "xlm-roberta", "longformer"}:
            model_limit -= self.model.config.pad_token_id + 1
        self.max_length = min(self.max_length, model_limit)
        self.special_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        if self.max_length <= self.special_count or self.tokenizer.sep_token_id is None:
            raise ValueError("Invalid maximum length or missing native separator")
        self.revision = revision
        self._forward_parameters = inspect.signature(self.model.forward).parameters

    def _pieces(self, query):
        encoded = self.tokenizer(list(query.tokens), is_split_into_words=True,
                                 add_special_tokens=False, truncation=False, verbose=False)
        parts = [[] for _ in query.tokens]
        for token, word_id in zip(encoded["input_ids"], encoded.word_ids()):
            if word_id is not None:
                parts[word_id].append(token)
        return parts

    def group_fits(self, queries):
        length = self.special_count + max(0, len(queries) - 1)
        for query in queries:
            pieces = self._pieces(query)
            if any(not part for part in pieces):
                return False
            length += sum(map(len, pieces))
        return length <= self.max_length

    def _wrap(self, body, mapping):
        probe = self.tokenizer.build_inputs_with_special_tokens([-987654321])
        position = probe.index(-987654321)
        prefix, suffix = probe[:position], probe[position + 1:]
        ids = self.tokenizer.build_inputs_with_special_tokens(body)
        if ids != prefix + body + suffix:
            raise ValueError("Unsupported tokenizer special-token layout")
        item = dict(input_ids=ids, attention_mask=[1] * len(ids))
        if "token_type_ids" in self.tokenizer.model_input_names:
            item["token_type_ids"] = self.tokenizer.create_token_type_ids_from_sequences(body)
        return item, [None] * len(prefix) + mapping + [None] * len(suffix)

    def _run(self, requests, hidden=False):
        outputs = []
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start:start + self.batch_size]
            prepared = [self._wrap(body, mapping) for body, mapping in chunk]
            inputs = self.tokenizer.pad([item for item, _ in prepared], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k in self._forward_parameters}
            if "global_attention_mask" in self._forward_parameters:
                mask = self.torch.zeros_like(inputs["input_ids"])
                mask[:, 0] = 1
                inputs["global_attention_mask"] = mask
            with self.torch.inference_mode():
                output = self.model(**inputs, output_hidden_states=hidden)
            values = output.hidden_states[-1] if hidden else output.logits
            values = values.float().cpu().numpy()
            for i, (_, mapping) in enumerate(prepared):
                outputs.append((values[i, :len(mapping)], mapping))
        return outputs

    def predict(self, queries):
        requests, owners = [], []
        for q in queries:
            chunks = chunk_word_pieces(self._pieces(q), self.max_length - self.special_count)
            for chunk in chunks:
                body, mapping = [], []
                for word, pieces in chunk:
                    body.extend(pieces)
                    mapping.extend([(q.id, word)] + [None] * (len(pieces) - 1))
                requests.append((body, mapping))
                owners.append(q.id)
        logits = {q.id: np.full((len(q.tokens), len(self.tags)), np.nan) for q in queries}
        for values, mapping in self._run(requests):
            for value, key in zip(values, mapping):
                if key is not None:
                    logits[key[0]][key[1]] = value
        return {q.id: Evidence.from_logits(logits[q.id], self.tags) for q in queries}

    def predict_groups(self, groups):
        requests = []
        for group in groups:
            body, mapping = [], []
            if not self.group_fits(group):
                raise ValueError("Group exceeds the input budget; pack it before inference")
            for q in group:
                if body:
                    body.append(self.tokenizer.sep_token_id)
                    mapping.append(None)
                for word, pieces in enumerate(self._pieces(q)):
                    body.extend(pieces)
                    mapping.extend([(q.id, word)] + [None] * (len(pieces) - 1))
            requests.append((body, mapping))
        outputs = []
        for group, (values, mapping) in zip(groups, self._run(requests)):
            logits = {q.id: np.full((len(q.tokens), len(self.tags)), np.nan) for q in group}
            for value, key in zip(values, mapping):
                if key is not None:
                    logits[key[0]][key[1]] = value
            outputs.append({q.id: Evidence.from_logits(logits[q.id], self.tags) for q in group})
        return outputs

    def _hidden_words(self, queries):
        requests, owners = [], []
        for q in queries:
            for chunk in chunk_word_pieces(self._pieces(q), self.max_length - self.special_count):
                body, mapping = [], []
                for word, pieces in chunk:
                    body.extend(pieces)
                    mapping.extend([(q.id, word)] + [None] * (len(pieces) - 1))
                requests.append((body, mapping))
                owners.append(q.id)
        pooled = {q.id: [] for q in queries}
        for values, mapping in self._run(requests, hidden=True):
            for value, key in zip(values, mapping):
                if key is not None:
                    pooled[key[0]].append(value)
        return {q.id: np.stack(pooled[q.id]) for q in queries}

    def embed(self, queries):
        pooled = self._hidden_words(queries)
        result = np.stack([pooled[q.id].mean(axis=0) for q in queries])
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)

    def embed_mentions(self, queries, mentions):
        words = self._hidden_words(queries)
        result = {}
        for q in queries:
            for span in mentions[q.id]:
                vector = words[q.id][span.start:span.end].mean(axis=0)
                result[(q.id, span.start, span.end)] = vector / max(np.linalg.norm(vector), 1e-12)
        return result

    def synchronize(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def describe(self):
        return {**super().describe(), "device": str(self.device), "batch_size": self.batch_size,
                "requested_revision": self.revision,
                "resolved_commit": getattr(self.model.config, "_commit_hash", None),
                "tokenizer": self.tokenizer.name_or_path,
                "word_reduction": "first_subtoken", "span_score": "mean_decoded_word_probability",
                "long_query_policy": "complete_word_chunks_no_dropped_words"}
