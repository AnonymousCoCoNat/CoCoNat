"""Offline correctness tests; no downloaded model or annotated benchmark is required."""

from dataclasses import fields
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from coconat.audit import export_audit, summarize_audit
from coconat.backends.external import load_source_manifest, parse_prediction_rows
from coconat.backends.hf import chunk_word_pieces
from coconat.backends.llm import HostedLLMBackend, parse_generation, select_demonstrations
from coconat.backends.toy import ToyBackend
from coconat.calibration import fit_temperature
from coconat.config import PipelineConfig, load_config
from coconat.data import example_from_dict, load_split, read_conll
from coconat.demo import make_demo
from coconat.io import redact_config, write_jsonl
from coconat.metrics import entity_ece, entity_metrics, jaccard, strata_summary, transitions
from coconat.pipeline import aggregate, detect, gradual_ordering, run_pipeline
from coconat.schema import Evidence, Example, Query, Span, softmax, spans_to_tags, tags_to_spans


class SchemaTests(unittest.TestCase):
    def test_query_excludes_gold(self):
        self.assertNotIn("gold", [f.name for f in fields(Query)])

    def test_span_validation(self):
        for args in [(-1, 2, "ORG"), (1, 1, "ORG"), (1, 2, "O"), (1, 2, "ORG", 2)]:
            with self.assertRaises(ValueError):
                Span(*args)

    def test_bio_multitoken_and_orphan(self):
        spans = tags_to_spans(["I-ORG", "I-ORG", "O", "B-PER", "B-PER"])
        self.assertEqual([s.key() for s in spans], [(0, 2, "ORG"), (3, 4, "PER"), (4, 5, "PER")])

    def test_bilou(self):
        spans = tags_to_spans(["B-ORG", "L-ORG", "U-PER", "O"])
        self.assertEqual([s.key() for s in spans], [(0, 2, "ORG"), (2, 3, "PER")])

    def test_bio_roundtrip(self):
        spans = (Span(1, 3, "LOC"), Span(3, 4, "ORG"))
        self.assertEqual(tags_to_spans(spans_to_tags(5, spans)), spans)

    def test_overlap_training_rejected(self):
        with self.assertRaises(ValueError):
            spans_to_tags(3, [Span(0, 2, "LOC"), Span(1, 3, "ORG")])

    def test_mean_span_confidence(self):
        evidence = Evidence.from_logits([[0, 3, 0], [0, 0, 2]], ("O", "B-ORG", "I-ORG"))
        expected = (softmax([[0, 3, 0]])[0, 1] + softmax([[0, 0, 2]])[0, 2]) / 2
        self.assertAlmostEqual(evidence.spans[0].score, expected)

    def test_softmax_stability_and_temperature(self):
        np.testing.assert_allclose(softmax([[1000, 1001]]).sum(), 1)
        with self.assertRaises(ValueError):
            softmax([[1, 2]], temperature=0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.queries = [Query("a", ("Acme", "arrived")), Query("b", ("Acme", "revenue")),
                        Query("c", ("London", "today"))]
        self.backend = ToyBackend()

    def test_zero_variance_and_empty_entities(self):
        prediction = {q.id: Evidence((Span(0, 1, "ORG", 0.8),)) for q in self.queries}
        result = detect(self.queries, prediction, PipelineConfig(kappa=0))
        self.assertEqual(result.low_spans, set())
        result = detect(self.queries, {q.id: Evidence(()) for q in self.queries}, PipelineConfig())
        self.assertIsNone(result.threshold)
        self.assertEqual(result.hard_queries, set())

    def test_top_two_not_full_distribution(self):
        qs = [Query(str(i), ("x",)) for i in range(10)]
        labels = ["ORG"] * 6 + ["PER"] * 3 + ["LOC"]
        first = {q.id: Evidence((Span(0, 1, label, .9),)) for q, label in zip(qs, labels)}
        d = detect(qs, first, PipelineConfig(delta=.7, detector="inconsistency"))
        self.assertAlmostEqual(d.ratios["x"], 2 / 3)
        self.assertEqual(len(d.hard_spans), 10)

    def test_affine_invariance(self):
        first = {q.id: Evidence((Span(0, 1, "ORG", score),))
                 for q, score in zip(self.queries, (.1, .5, .9))}
        other = {qid: Evidence(tuple(Span(s.start, s.end, s.label, .5 * s.score + .2) for s in ev.spans))
                 for qid, ev in first.items()}
        config = PipelineConfig(kappa=1, detector="confidence")
        self.assertEqual(detect(self.queries, first, config).hard_spans,
                         detect(self.queries, other, config).hard_spans)

    def test_case_sensitive_surface(self):
        qs = [Query("a", ("Apple",)), Query("b", ("apple",))]
        first = {"a": Evidence((Span(0, 1, "ORG", .9),)), "b": Evidence((Span(0, 1, "PER", .9),))}
        self.assertEqual(detect(qs, first, PipelineConfig()).hard_spans, set())
        self.assertEqual(len(detect(qs, first, PipelineConfig(case_sensitive=False)).hard_spans), 2)

    def test_nonhard_frozen(self):
        result = run_pipeline(self.queries, self.backend, PipelineConfig(detector="inconsistency"))
        self.assertEqual(result.first["c"].spans, result.final["c"].spans)
        self.assertNotIn("c", result.processed)
        self.assertEqual(result.final["a"].spans[0].label, "ORG")

    def test_singleton_fallback(self):
        result = run_pipeline([self.queries[0]], self.backend, PipelineConfig(kappa=0))
        self.assertFalse(result.processed)
        self.assertEqual(result.first["a"].spans, result.final["a"].spans)

    def test_budget_fallback(self):
        backend = ToyBackend(max_length=3)
        result = run_pipeline(self.queries, backend, PipelineConfig())
        self.assertFalse(result.processed)
        self.assertTrue(all(result.first[k].spans == result.final[k].spans for k in result.first))

    def test_none_means_backbone(self):
        result = run_pipeline(self.queries, self.backend, PipelineConfig(detector="none"))
        self.assertFalse(result.processed)

    def test_gradual_ordering_is_permutation_and_separates_anchors(self):
        records = [(str(i), f"L{i % 3}", .4 + i / 20) for i in range(10)]
        result = gradual_ordering(records)
        self.assertEqual(sorted(result), sorted(records))
        self.assertIn(records[-1], [result[0], result[-1]])
        self.assertIn(records[-2], [result[0], result[-1]])
        self.assertEqual(result, gradual_ordering(list(reversed(records))))

    def test_mean_uses_all_contexts(self):
        # A typed span is scored even in a context whose argmax favors another type.
        class Fixed(Evidence):
            def scores_for_boundary(self, start, end, labels):
                return self.metadata
        evidences = [Fixed((Span(0, 1, "ORG"),), metadata=v) for v in
                     [{"O": 0., "PER": .8, "ORG": .1, "LOC": .1},
                      {"O": 0., "PER": .1, "ORG": .7, "LOC": .2},
                      {"O": 0., "PER": .15, "ORG": .55, "LOC": .3}]]
        self.assertEqual(aggregate(evidences, ("ORG", "PER", "LOC"), "mean").spans[0].label, "ORG")
        self.assertEqual(aggregate(evidences, ("ORG", "PER", "LOC"), "max").spans[0].label, "PER")
        self.assertAlmostEqual(aggregate(evidences, ("ORG", "PER", "LOC"), "mean").spans[0].score, .45)

    def test_missing_context_is_not_dropped_from_mean(self):
        output = aggregate([Evidence((Span(0, 1, "ORG", .9),)), Evidence(())], ("ORG",), "mean")
        self.assertEqual(output.spans, ())

    def test_config_rejects_typos(self):
        with self.assertRaises(ValueError):
            PipelineConfig.from_dict({"kapppa": 9})
        with self.assertRaises(ValueError):
            PipelineConfig(delta=1.1)

    def test_whole_word_chunking(self):
        chunks = chunk_word_pieces([[1, 2], [3], [4, 5]], 3)
        self.assertEqual([[i for i, _ in c] for c in chunks], [[0, 1], [2]])
        with self.assertRaises(ValueError):
            chunk_word_pieces([[1, 2, 3]], 2)


class MetricTests(unittest.TestCase):
    def test_exact_boundary_and_type(self):
        ex = Example(Query("q", ("New", "York")), (Span(0, 2, "LOC"),))
        scores = entity_metrics([ex], {"q": Evidence((Span(1, 2, "LOC"),))})
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (0, 1, 1))

    def test_ece_equal_frequency(self):
        ex = Example(Query("q", ("a", "b", "c", "d")), (Span(0, 1, "ORG"), Span(2, 3, "ORG")))
        prediction = Evidence(tuple(Span(i, i + 1, "ORG", .8) for i in range(4)))
        ece = entity_ece([ex], {"q": prediction}, n_bins=1)
        self.assertAlmostEqual(ece["ece"], .3)
        self.assertEqual(sum(b["n"] for b in entity_ece([ex], {"q": prediction}, 3)["bins"]), 4)

    def test_all_four_transitions(self):
        ex = Example(Query("q", tuple("abcdefghij")), tuple(Span(i, i + 1, "ORG") for i in (0, 2, 6, 8)))
        before = Evidence(tuple(Span(i, i + 1, label) for i, label in [(0, "PER"), (2, "ORG"), (4, "PER"), (6, "ORG")]))
        after = Evidence(tuple(Span(i, i + 1, label) for i, label in [(0, "ORG"), (2, "ORG"), (4, "LOC"), (6, "PER"), (8, "ORG")]))
        result = run_pipeline([ex.query], ToyBackend(), PipelineConfig(detector="none"), first={"q": before})
        result.final = {"q": after}
        result.processed = {"q"}
        table, cases = transitions([ex], result)
        self.assertEqual(table["counts"], {"W_C": 1, "C_C": 1, "W_W": 1, "C_W": 1})
        self.assertEqual(table["net_correction_fraction"], 0)
        self.assertEqual(table["new_only_second_spans"], {"total": 1, "correct": 1, "wrong": 0})
        self.assertEqual(table["n"], 4)
        self.assertEqual(len(cases), 4)

    def test_transition_alignment_follows_each_first_span(self):
        ex = Example(Query("q", ("New", "York")), (Span(0, 2, "LOC"),))
        before = Evidence((Span(0, 1, "ORG"), Span(1, 2, "ORG")))
        after = Evidence((Span(0, 2, "LOC"),))
        result = run_pipeline([ex.query], ToyBackend(), PipelineConfig(detector="none"), first={"q": before})
        result.final, result.processed = {"q": after}, {"q"}
        table, _ = transitions([ex], result)
        self.assertEqual(table["counts"]["W_C"], 2)
        self.assertEqual(table["one_to_one_greedy_counts"]["W_C"], 1)
        self.assertEqual(table["one_to_one_greedy_counts"]["W_W"], 1)

    def test_calibration_cannot_use_test(self):
        with self.assertRaises(ValueError):
            fit_temperature([], {}, split="test")

    def test_temperature_nll_decreases(self):
        ex = Example(Query("q", ("a", "b")), (Span(0, 1, "ORG"),))
        raw = Evidence.from_logits([[0, 4, -2], [0, 4, -2]], ("O", "B-ORG", "I-ORG"))
        fit = fit_temperature([ex], {"q": raw})
        self.assertLessEqual(fit["calibrated_nll"], fit["raw_nll"])
        self.assertEqual(raw.spans, raw.with_temperature(fit["temperature"]).spans)

    def test_empty_jaccard(self):
        self.assertEqual(jaccard(set(), set()), 1)

    def test_stratum_denominator(self):
        rows = [dict(label="ORG", confidence=.8, low_flag=True, wrong=True),
                dict(label="ORG", confidence=.9, low_flag=False, wrong=False)]
        result = strata_summary(rows, "label")[0]
        self.assertEqual(result["flagged_fraction"], .5)
        self.assertEqual(result["error_among_flagged"], 1)


class InputTests(unittest.TestCase):
    def test_conll_documents_and_tag_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.conll"
            path.write_text("-DOCSTART- O\n\nNew B-LOC\nYork I-LOC\n\n-DOCSTART- O\n\nAcme B-ORG\n")
            examples = read_conll(path)
            self.assertEqual(examples[0].gold, (Span(0, 2, "LOC"),))
            self.assertNotEqual(examples[0].query.doc_id, examples[1].query.doc_id)

    def test_demo_data_and_config(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_config(make_demo(temp))
            for split in ("train", "validation", "test"):
                self.assertTrue(load_split(config["dataset"], split))

    def test_token_tag_length(self):
        with self.assertRaises(ValueError):
            example_from_dict(dict(tokens=["a", "b"], ner_tags=["O"]), "q")

    def test_external_coverage_and_offsets(self):
        q = Query("q", ("Acme",))
        with self.assertRaises(ValueError):
            parse_prediction_rows([q], [], ["ORG"])
        with self.assertRaises(ValueError):
            parse_prediction_rows([q], [dict(id="q", tokens=["other"], spans=[])], ["ORG"])
        result = parse_prediction_rows([q], [dict(id="q", tokens=["Acme"], spans=[dict(start=0, end=1, label="ORG")])], ["ORG"])
        self.assertFalse(result["q"].confidence_available)

    def test_provenance_template_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.json"
            path.write_text(json.dumps(dict(method="X", source_url="https://example.org",
                source_revision="REPLACE_WITH_COMMIT", checkpoint="model", supervision="zero-shot")))
            with self.assertRaises(ValueError):
                load_source_manifest(path)

    def test_external_prediction_baseline_end_to_end(self):
        from coconat.runner import run_suite

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = load_config(make_demo(root / "fixture"))
            examples = load_split(config["dataset"], "test")
            predicted = ToyBackend().predict([ex.query for ex in examples])
            prediction_path = root / "predictions.jsonl"
            write_jsonl(prediction_path, [dict(id=ex.query.id, tokens=list(ex.query.tokens),
                spans=[span.to_dict() for span in predicted[ex.query.id].spans]) for ex in examples])
            source = root / "source.json"
            source.write_text(json.dumps(dict(method="VerifiedExample", source_url="https://example.org/code",
                source_revision="abc123", checkpoint="checkpoint-v1", supervision="test fixture")))
            config["baselines"] = [dict(name="VerifiedExample", kind="external_predictions",
                family="test-only", regime="test fixture", enabled=True,
                prediction_path=str(prediction_path), source_manifest=str(source))]
            config["required_baselines"] = ["VerifiedExample"]
            run = run_suite(config, tasks=("baselines",), output=root / "runs", require_all=True)
            summary = json.loads((run / "summary.json").read_text())
            self.assertIsNone(summary["baselines"][0]["seconds_per_1000"])
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "completed")

    def test_generation_not_gold_repaired(self):
        q = Query("q", ("Acme", "arrived"))
        response = '```json\n{"entities":[{"start":0,"end":1,"label":"ORG"},{"start":8,"end":9,"label":"ORG"}]}\n```'
        ev = parse_generation(response, q, ["ORG"])
        self.assertEqual(len(ev.spans), 1)
        self.assertEqual(ev.metadata["invalid_entities"], 1)
        self.assertTrue(parse_generation("not JSON", q, ["ORG"]).metadata["parse_failed"])

    def test_hosted_calls_require_opt_in(self):
        with self.assertRaises(PermissionError):
            HostedLLMBackend("anything", ["ORG"])

    def test_hosted_payload_and_response_without_network(self):
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content":
            '{"entities":[{"start":0,"end":1,"label":"ORG"}]}'}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 20}, "model": "snapshot"}).encode())
        with mock.patch.dict(os.environ, {"TEST_OPENAI_KEY": "test-only"}), \
                mock.patch("urllib.request.urlopen", return_value=response) as opened:
            backend = HostedLLMBackend("snapshot", ["ORG"], allow_api=True,
                                       api_key_env="TEST_OPENAI_KEY")
            result = backend.predict([Query("q", ("Acme",))])
        request = opened.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(result["q"].spans, (Span(0, 1, "ORG"),))

    def test_demonstrations_come_from_train(self):
        train = [Example(Query(str(i), ("Acme",)), (Span(0, 1, "ORG"),)) for i in range(6)]
        demos = select_demonstrations(train, ["ORG"], 5)
        self.assertEqual(len(demos), 5)
        self.assertTrue(all(d in train for d in demos))

    def test_redaction(self):
        self.assertEqual(redact_config({"api_key": "secret"})["api_key"], "[REDACTED]")

    def test_manual_audit_remains_unannotated(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "cases.jsonl", Path(temp) / "audit.csv"
            write_jsonl(source, [dict(transition="C_W", query_id="q", text="Acme", before={}, after={}, gold=[])])
            self.assertEqual(export_audit([source], output), 1)
            summary = Path(temp) / "summary.json"
            summarize_audit(output, summary)
            self.assertFalse(json.loads(summary.read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
