"""Optional native span adapters. Label maps must be declared, not guessed."""

from __future__ import annotations

from .base import Backend, char_span_to_word
from ..schema import Evidence, Span


class GLiNERBackend(Backend):
    def __init__(self, checkpoint, labels, label_map=None, threshold=0.5, device="cpu",
                 max_length=512, **kwargs):
        from gliner import GLiNER

        self.name, self.labels = checkpoint, tuple(labels)
        self.device = "cpu" if device == "auto" else device
        self.model = GLiNER.from_pretrained(checkpoint).to(self.device)
        self.label_map = label_map or {label: label for label in labels}
        if set(self.label_map.values()) != set(self.labels):
            raise ValueError("GLiNER label_map must cover all dataset types exactly")
        self.threshold = float(threshold)
        self.max_length = int(max_length)

    def predict(self, queries):
        output = {}
        for q in queries:
            entities = self.model.predict_entities(q.text, list(self.label_map), threshold=self.threshold)
            spans, rejected = [], 0
            for entity in entities:
                boundary = char_span_to_word(q, entity["start"], entity["end"])
                if boundary is None:
                    rejected += 1
                    continue
                spans.append(Span(*boundary, self.label_map[entity["label"]], float(entity["score"])))
            output[q.id] = Evidence(tuple(sorted(set(spans))), metadata={"invalid_offsets": rejected})
        return output

    def synchronize(self):
        if str(self.device).startswith("cuda"):
            import torch
            torch.cuda.synchronize()


class FlairBackend(Backend):
    def __init__(self, checkpoint, labels, label_map=None, device="cpu", max_length=512, **kwargs):
        import flair
        import torch
        from flair.models import SequenceTagger

        self.device = torch.device("cpu" if device == "auto" else device)
        flair.device = self.device
        self.model = SequenceTagger.load(checkpoint)
        self.name, self.labels = checkpoint, tuple(labels)
        self.label_map = label_map or {label: label for label in labels}
        self.max_length = int(max_length)

    def predict(self, queries):
        from flair.data import Sentence

        sentences = [Sentence(list(q.tokens), use_tokenizer=False) for q in queries]
        self.model.predict(sentences)
        output = {}
        for q, sentence in zip(queries, sentences):
            spans = []
            for entity in sentence.get_spans("ner"):
                annotation = entity.get_label("ner")
                if annotation.value not in self.label_map:
                    raise ValueError(f"No mapping for Flair type {annotation.value}")
                spans.append(Span(entity[0].idx - 1, entity[-1].idx,
                                  self.label_map[annotation.value], annotation.score))
            output[q.id] = Evidence(tuple(spans))
        return output

    def synchronize(self):
        import torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class SpacyBackend(Backend):
    def __init__(self, checkpoint, labels, label_map=None, spans_key="sc", max_length=512, **kwargs):
        import spacy

        self.model = spacy.load(checkpoint)
        self.name, self.labels = checkpoint, tuple(labels)
        self.label_map = label_map or {label: label for label in labels}
        self.spans_key = spans_key
        self.max_length = int(max_length)

    def predict(self, queries):
        from spacy.tokens import Doc

        output = {}
        for q in queries:
            doc = Doc(self.model.vocab, words=list(q.tokens))
            for _, component in self.model.pipeline:
                doc = component(doc)
            native = doc.spans.get(self.spans_key)
            if native is None or "scores" not in native.attrs:
                raise ValueError("Use a trained spaCy SpanCategorizer with span scores, not bare doc.ents")
            spans = []
            for entity, score in zip(native, native.attrs["scores"]):
                if entity.label_ not in self.label_map:
                    raise ValueError(f"No mapping for spaCy type {entity.label_}")
                spans.append(Span(entity.start, entity.end, self.label_map[entity.label_], float(score)))
            output[q.id] = Evidence(tuple(spans))
        return output
