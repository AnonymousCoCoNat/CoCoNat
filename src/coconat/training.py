"""Optional supervised BIO training; CoCoNat itself never updates model weights."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .data import load_split
from .io import environment_info, labeled_fingerprint, write_json
from .schema import spans_to_tags
from .backends.hf import chunk_word_pieces


def _training_rows(examples, tokenizer, label2id, max_length, context_mode):
    collections = []
    if context_mode == "document":
        groups = defaultdict(list)
        for ex in examples:
            if ex.query.doc_id is None:
                raise ValueError("Document-context training requires document IDs")
            groups[ex.query.doc_id].append(ex)
        collections = list(groups.values())
    elif context_mode == "sentence":
        collections = [[ex] for ex in examples]
    else:
        raise ValueError("training.context_mode must be sentence or document")
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    rows = []
    for collection in collections:
        pieces, word_labels = [], []
        for ex in collection:
            if pieces:
                pieces.append([tokenizer.sep_token_id])
                word_labels.append(-100)
            encoded = tokenizer(list(ex.query.tokens), is_split_into_words=True,
                                truncation=False, add_special_tokens=False)
            local = [[] for _ in ex.query.tokens]
            for token, word in zip(encoded["input_ids"], encoded.word_ids()):
                if word is not None:
                    local[word].append(token)
            pieces.extend(local)
            word_labels.extend(label2id[t] for t in spans_to_tags(len(ex.query.tokens), ex.gold))
        for chunk in chunk_word_pieces(pieces, budget):
            body, targets = [], []
            for word, word_pieces in chunk:
                body.extend(word_pieces)
                targets.extend([word_labels[word]] + [-100] * (len(word_pieces) - 1))
            probe = tokenizer.build_inputs_with_special_tokens([-987654321])
            pos = probe.index(-987654321)
            prefix, suffix = probe[:pos], probe[pos + 1:]
            item = dict(input_ids=tokenizer.build_inputs_with_special_tokens(body),
                        labels=[-100] * len(prefix) + targets + [-100] * len(suffix))
            item["attention_mask"] = [1] * len(item["input_ids"])
            if "token_type_ids" in tokenizer.model_input_names:
                item["token_type_ids"] = tokenizer.create_token_type_ids_from_sequences(body)
            rows.append(item)
    return rows


def train(config, output=None):
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed)

    settings = config.get("training", {})
    seed = int(settings.get("seed", 42))
    set_seed(seed)
    train_data = load_split(config["dataset"], "train")
    validation = load_split(config["dataset"], "validation")
    if labeled_fingerprint(train_data) == labeled_fingerprint(validation):
        raise ValueError("Training and validation data are identical")
    destination = Path(output or config["model"]["checkpoint"])
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {destination}")
    base = settings.get("base_checkpoint", "microsoft/deberta-v3-base")
    labels = config["dataset"]["labels"]
    tags = ["O"] + [tag for label in labels for tag in ("B-" + label, "I-" + label)]
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=False,
                                              revision=settings.get("revision"))
    if not tokenizer.is_fast:
        raise ValueError("Training requires a fast tokenizer")
    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(tags), id2label=dict(enumerate(tags)),
        label2id={tag: i for i, tag in enumerate(tags)}, trust_remote_code=False,
        revision=settings.get("revision"), ignore_mismatched_sizes=True)
    length = int(config["model"].get("max_length", 512))
    max_positions = getattr(model.config, "max_position_embeddings", length)
    if getattr(model.config, "model_type", "") in {"roberta", "xlm-roberta", "longformer"}:
        max_positions -= model.config.pad_token_id + 1
    length = min(length, int(tokenizer.model_max_length), max_positions)
    mode = settings.get("context_mode", "sentence")
    train_rows = _training_rows(train_data, tokenizer, model.config.label2id, length, mode)
    val_rows = _training_rows(validation, tokenizer, model.config.label2id, length, mode)
    collator = DataCollatorForTokenClassification(tokenizer)
    if getattr(model.config, "model_type", "") == "longformer":
        base_collator = collator

        def collator(features):
            batch = base_collator(features)
            batch["global_attention_mask"] = batch["attention_mask"].new_zeros(batch["attention_mask"].shape)
            batch["global_attention_mask"][:, 0] = 1
            return batch

    args = TrainingArguments(
        output_dir=str(destination / "training_state"), eval_strategy="epoch", save_strategy="epoch",
        learning_rate=float(settings.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(settings.get("batch_size", 8)),
        per_device_eval_batch_size=int(settings.get("batch_size", 8)),
        num_train_epochs=float(settings.get("epochs", 3)), weight_decay=0.01,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, seed=seed, data_seed=seed, report_to=[],
        use_cpu=bool(settings.get("use_cpu", False)), dataloader_num_workers=0,
        gradient_accumulation_steps=int(settings.get("gradient_accumulation_steps", 1)),
        fp16=bool(settings.get("fp16", False)), bf16=bool(settings.get("bf16", False)))
    trainer = Trainer(model=model, args=args, train_dataset=train_rows, eval_dataset=val_rows,
                      data_collator=collator, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))
    write_json(destination / "training_manifest.json", dict(
        base_checkpoint=base, settings=settings, max_length=length,
        train_sha256=labeled_fingerprint(train_data), validation_sha256=labeled_fingerprint(validation),
        labels=tags, selection="minimum validation loss over completed epochs", environment=environment_info()))
    return destination
