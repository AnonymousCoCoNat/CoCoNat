"""Real CPU Transformer tests with tiny random local weights, never benchmark results."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from coconat.backends.hf import HFBackend
from coconat.schema import Example, Query, Span


HF_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HF_AVAILABLE, "Install .[hf] to run actual Transformer integration tests")
class TinyHFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.model_path = cls.root / "tiny-bert"
        cls.model_path.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Acme", "Amazon", "river", "arrived", ".", "in", "London"]
        vocab_file = cls.root / "vocab.txt"
        vocab_file.write_text("\n".join(vocab) + "\n")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False, model_max_length=32)
        tokenizer.save_pretrained(cls.model_path)
        tags = ["O", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
                            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
                            num_labels=len(tags), id2label=dict(enumerate(tags)),
                            label2id={tag: i for i, tag in enumerate(tags)})
        BertForTokenClassification(config).save_pretrained(cls.model_path)
        cls.backend = HFBackend(str(cls.model_path), ["ORG", "LOC"], max_length=16, device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_single_sentence_logits_and_offsets(self):
        q = Query("q", ("Acme", "arrived", "in", "London", "."))
        result = self.backend.predict([q])["q"]
        self.assertEqual(result.logits.shape, (5, 5))
        self.assertTrue(np.isfinite(result.logits).all())

    def test_long_query_keeps_all_words(self):
        q = Query("long", ("Acme",) * 80)
        result = self.backend.predict([q])["long"]
        self.assertEqual(result.logits.shape, (80, 5))
        self.assertTrue(np.isfinite(result.logits).all())

    def test_group_separator_does_not_shift_offsets(self):
        group = [Query("a", ("Acme", "arrived")), Query("b", ("Amazon", "river", "."))]
        result = self.backend.predict_groups([group])[0]
        self.assertEqual(result["a"].logits.shape, (2, 5))
        self.assertEqual(result["b"].logits.shape, (3, 5))
        for q in group:
            self.assertTrue(all(0 <= s.start < s.end <= len(q.tokens) for s in result[q.id].spans))

    def test_over_budget_group_rejected(self):
        with self.assertRaises(ValueError):
            self.backend.predict_groups([[Query("q", ("Acme",) * 80)]])

    def test_real_hidden_embeddings(self):
        qs = [Query("a", ("Acme",)), Query("b", ("Amazon",))]
        embeddings = self.backend.embed(qs)
        self.assertEqual(embeddings.shape, (2, 16))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)

    def test_training_rows_ignore_special_and_subword_tokens(self):
        from coconat.training import _training_rows

        examples = [Example(Query("q", ("Acme", "arrived")), (Span(0, 1, "ORG"),))]
        rows = _training_rows(examples, self.backend.tokenizer, self.backend.model.config.label2id, 16, "sentence")
        self.assertEqual(rows[0]["labels"], [-100, 1, 0, -100])

    def test_actual_train_save_reload(self):
        from coconat.config import load_config
        from coconat.demo import make_demo
        from coconat.runner import run_suite
        from coconat.training import train

        config = load_config(make_demo(self.root / "training-fixture"))
        config["model"] = dict(kind="hf", checkpoint=str(self.root / "trained"), max_length=16)
        config["training"] = dict(base_checkpoint=str(self.model_path), epochs=1, batch_size=4,
                                  learning_rate=1e-4, use_cpu=True)
        destination = train(config)
        self.assertTrue((destination / "config.json").is_file())
        backend = HFBackend(str(destination), config["dataset"]["labels"], max_length=16, device="cpu")
        result = backend.predict([Query("trained-query", ("Acme", "arrived"))])
        self.assertEqual(result["trained-query"].logits.shape, (2, 7))
        config["model"].update(device="cpu", batch_size=4)
        config["baselines"] = [dict(kind="toy", name="test-only-toy", family="test-only",
                                     regime="synthetic", enabled=True)]
        run = run_suite(config, output=self.root / "actual-suite", require_all=True)
        self.assertTrue((run / "report" / "sensitivity.pdf").is_file())
        self.assertTrue((run / "main" / "transitions.json").is_file())

    def test_document_context_baseline(self):
        from coconat.experiments import benchmark_backend

        examples = [Example(Query("a", ("Acme",), "d"), (Span(0, 1, "ORG"),)),
                    Example(Query("b", ("London",), "d"), (Span(0, 1, "LOC"),))]
        prediction, timing = benchmark_backend(examples, self.backend, repeats=1, warmups=0, document_mode=True)
        self.assertEqual(set(prediction), {"a", "b"})
        self.assertTrue(timing["document_context"])

    def test_longformer_local_weights(self):
        from transformers import LongformerConfig, LongformerForTokenClassification

        path = self.root / "tiny-longformer"
        self.backend.tokenizer.save_pretrained(path)
        tags = self.backend.tags
        config = LongformerConfig(vocab_size=len(self.backend.tokenizer), hidden_size=16,
             num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
             attention_window=[4], max_position_embeddings=34, pad_token_id=0,
             num_labels=len(tags), id2label=dict(enumerate(tags)),
             label2id={tag: i for i, tag in enumerate(tags)})
        LongformerForTokenClassification(config).save_pretrained(path)
        backend = HFBackend(str(path), ["ORG", "LOC"], max_length=16, device="cpu")
        result = backend.predict_groups([[Query("a", ("Acme",)), Query("b", ("Amazon", "river"))]])
        self.assertEqual(result[0]["b"].logits.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
