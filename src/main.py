# main.py
"""
CoCoNat: Leveraging Cross-Query Context to Enhance Named Entity Recognition

This script reproduces the collective NER pipeline described in our paper.
It first runs a pre‑trained NER backbone on every query ("first pass"),
identifies hard queries, forms context‑coherent groups, concatenates each
group into a single block and performs a second NER pass. Finally, span‑level
predictions from multiple groups are fused via a configurable aggregation rule.
The pipeline is organised into three stages that mirror Sections 4.1–4.3 of our paper:
    1. Hard‑query detection (§4.1).
    2. Query grouping (§4.2).
    3. Second‑pass tagging and span‑level aggregation (§4.3).
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Set

from datasets import load_dataset
import evaluate
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from utils import *

###############################################################################
#                               Hyperparameters                               #
###############################################################################
KAPPA: float = 8.0          # Low‑confidence threshold
DELTA: float = 0.9          # Inconsistency threshold
AGGREGATION: str = "MEAN"   # {MEAN, MAX, PRODUCT}

MAX_GROUP_SIZE: int = 10    # Maximum queries per group after sub‑division
MIN_GROUP_SIZE: int = 1     # Minimum queries per group to keep
INCONS_TOLERANCE: int = 2   # Required number of conflicting labels to trigger
UPPER_BOUND: int = 80       # Ignore entities that occur extremely often
LOWER_BOUND: int = 0        # Ignore entities that occur too rarely
ADD_SEPARATOR: bool = True  # Insert a '.' delimiter between queries
CASED: bool = False         # If True, grouping is performed case‑sensitively
###############################################################################
#                            Model & dataset paths                            #
###############################################################################
TRAIN: bool = True  # Toggle train/test
DATASET_NAME = "conll2003"
DATASET_LOC = "conll2003"
MODEL_NAME = "DeBERTa-v3-base"
MODEL_CHECKPOINT = "Microsoft/DeBERTa-v3-base"
LOCAL_CKPT = f"./output/{DATASET_NAME}/{MODEL_NAME}/checkpoint-5000"
LOCAL_FOLDER = f"./output/{DATASET_NAME}"
LOCAL_FOLDER_MODEL = f"{LOCAL_FOLDER}/{MODEL_NAME}"
###############################################################################
#                          Dataset & backbone loading                         #
###############################################################################
print("Loading dataset...")
dataset = load_dataset(DATASET_LOC, trust_remote_code=True, cache_dir=LOCAL_FOLDER)
LABEL_NAMES = dataset["train"].features["ner_tags"].feature.names
ID2LABEL = dict(enumerate(LABEL_NAMES))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

print("Loading model...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT if TRAIN else LOCAL_CKPT,
    num_labels=len(ID2LABEL),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, model_max_length=512)


def tokenize_and_align_labels(examples: dict) -> dict:
    """Tokenize examples and propagate word‑level tags to sub‑tokens."""
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )
    aligned_labels = []
    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        lbl_ids = []
        for w in word_ids:
            if w is None:
                lbl_ids.append(-100)
            elif w != prev:
                lbl_ids.append(ner_tags[w])
            else:
                lbl_ids.append(ner_tags[w])
            prev = w
        aligned_labels.append(lbl_ids)
    tokenized["labels"] = aligned_labels
    return tokenized


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = tokenized_dataset["test"]


def compute_metrics(eval_tuple):
    """SeqEval wrapper for the HuggingFace Trainer."""
    metric = evaluate.load("seqeval")
    logits, labels = eval_tuple
    preds = np.argmax(logits, axis=2)

    p_labels, t_labels = [], []
    for p, l in zip(preds, labels):
        p_clean, l_clean = [], []
        for pid, lid in zip(p, l):
            if lid != -100:
                p_clean.append(ID2LABEL[pid])
                l_clean.append(ID2LABEL[lid])
        p_labels.append(p_clean)
        t_labels.append(l_clean)
    res = metric.compute(predictions=p_labels, references=t_labels)
    return {k.replace("overall_", ""): v for k, v in res.items()}


training_args = TrainingArguments(
    output_dir=LOCAL_FOLDER_MODEL,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

if TRAIN:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        eval_dataset=tokenized_dataset["test"],
    )


def process_group(
    subset_input_ids: List[List[int]],
    groups: List[List[int]],
    ner: torch.nn.Module,
    device: torch.device,
    max_length: int = 512,
) -> List[Dict[int, np.ndarray]]:
    """Run the backbone once per group batch and slice logits per query.

    Each group is a list of query indices. We concatenate the queries inside
    the group, optionally inserting a period token between them, forward the
    sequence through ner, then slice the resulting logits back into the
    original per‑query segments.
    """
    CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    DOT = tokenizer.convert_tokens_to_ids('.')

    batch_ids, offsets_per_group = [], []
    for g in groups:
        concat: List[int] = []
        offsets = [0]
        for q in g:
            body = subset_input_ids[q][1:-1]  # strip CLS/SEP
            concat.extend(body)
            if ADD_SEPARATOR:
                concat.append(DOT)
            offsets.append(len(concat))
        seq = [CLS] + concat[: max_length - 2] + [SEP]
        batch_ids.append(seq)
        offsets_per_group.append(offsets)

    # Pad batch
    max_len = max(len(s) for s in batch_ids)
    ids_tensor = torch.tensor([s + [PAD] * (max_len - len(s)) for s in batch_ids], device=device)
    mask_tensor = torch.tensor([[1] * len(s) + [0] * (max_len - len(s)) for s in batch_ids], device=device)

    with torch.no_grad():
        logits = ner(input_ids=ids_tensor, attention_mask=mask_tensor).logits.cpu().numpy()

    outputs: List[Dict[int, np.ndarray]] = []
    for b_idx, g in enumerate(groups):
        slc = {}
        offs = offsets_per_group[b_idx]
        for i, q in enumerate(g):
            s, e = offs[i] + 1, offs[i + 1] + 1  # skip CLS
            slc[q] = logits[b_idx][s:e]
        outputs.append(slc)
    return outputs


def aggregate_word_predictions(
    tokens: List[str],
    logits: np.ndarray
) -> List[Dict[str, str | float]]:
    """Convert sub‑token logits to span‑level predictions."""
    mapping = tokenizer(tokens, is_split_into_words=True).word_ids()
    spans, cur = [], None
    for i, w in enumerate(mapping):
        if w is None:
            continue
        if cur is None:
            cur = (w, [i])
        elif w == cur[0]:
            cur[1].append(i)
        else:
            spans.append(cur)
            cur = (w, [i])
    if cur:
        spans.append(cur)

    results = []
    for w_id, idxs in spans:
        probs = np.vstack([np.exp(l) / np.exp(l).sum() for l in logits[idxs]])
        argmaxes = probs[np.arange(len(probs)), probs.argmax(1)]
        best_row = argmaxes.argmax()
        best_lab = ID2LABEL[probs[best_row].argmax()]
        results.append({"text": tokens[w_id], "label": best_lab, "score": argmaxes[best_row]})
    return results


def hard_query_detector(
    queries_with_entities: List[Tuple[int, List[dict]]],
    threshold_l: float,
    threshold_c: float,
) -> Tuple[List[Tuple[int, List[dict]]], Set[str]]:
    """
    Identify hard entities and return the queries that contain them.

    Criteria
    --------
    1. Low confidence: entity probability below threshold_l.
    2. Inconsistency: the same surface form is assigned >=2 labels and the
       most frequent label accounts for <= threshold_c of the occurrences.
    """
    # Collect statistics
    label_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    low_conf: Set[str] = set()
    for _, ents in queries_with_entities:
        for ent in ents:
            form, lab, conf = ent.values()
            label_counts.setdefault(form, defaultdict(int))[lab] += 1
            if conf < threshold_l:
                low_conf.add(form)

    inconsistent: Set[str] = set()
    for form, counts in label_counts.items():
        if len(counts) >= INCONS_TOLERANCE:
            top, second = sorted(counts.values(), reverse=True)[:2]
            if top / (top + second) <= threshold_c:
                inconsistent.add(form)

    hard_entities = (low_conf | inconsistent) - {p for p in low_conf | inconsistent if
                                                 (sum(label_counts[p].values()) <= LOWER_BOUND or
                                                  sum(label_counts[p].values()) > UPPER_BOUND or
                                                  p.replace(",", "").replace(".", "").isdigit())}
    punctuations = {'~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
                    ',', '.', '"', "'", '/', '-', ':', ';', '+', '=', '_',
                    '<', '>', '?', '|', '{', '}', '[', ']', "`", '--'}
    hard_entities -= punctuations

    hard_queries = [(idx, ents) for idx, ents in queries_with_entities
                    if hard_entities & {e["text"] for e in ents}]
    return hard_queries, hard_entities


def cluster_queries(
    queries_with_entities: List[Tuple[int, List[dict]]],
    hard_entities: Set[str],
) -> Dict[str, List[int]]:
    """Form initial groups anchored by surface‑matched hard entities."""
    clusters: Dict[str, List[int]] = {e.lower() if not CASED else e: [] for e in hard_entities}
    for idx, ents in queries_with_entities:
        ents_in_q = {e["text"] for e in ents} if CASED else {e["text"].lower() for e in ents}
        for ent in ents_in_q & clusters.keys():
            clusters[ent].append(idx)
    return {k: v for k, v in clusters.items() if len(v) >= MIN_GROUP_SIZE}


def main():
    """Execute the full CoCoNat pipeline on the test split."""

    device = next(model.parameters()).device

    # ---------------- First pass -------------------------------------------
    first_logits = trainer.predict(test_dataset).predictions

    queries_with_entities: List[Tuple[int, List[dict]]] = []
    naive_preds: Dict[int, List[str]] = {}
    tokens_list = test_dataset["tokens"]
    for idx, (toks, log) in enumerate(zip(tokens_list, first_logits)):
        ents = aggregate_word_predictions(toks, log)
        queries_with_entities.append((idx, ents))
        naive_preds[idx] = [e["label"] for e in ents]

    # ---------------- Detect hard queries ----------------------------------
    conf_thresh, _, _ = auto_threshold(queries_with_entities, KAPPA)
    hard_qs, hard_ents = hard_query_detector(queries_with_entities, conf_thresh, DELTA)

    # ---------------- Grouping & ordering ----------------------------------
    init_groups = cluster_queries(hard_qs, hard_ents)
    subdivided = subdivide_groups(init_groups)
    all_groups = list(subdivided.values())

    # ---------------- Second pass ------------------------------------------
    batch_size = 16
    model.eval()
    second_logits: Dict[int, np.ndarray] = {}
    for start in tqdm(range(0, len(all_groups), batch_size)):
        batch = all_groups[start:start + batch_size]
        batch_pred = process_group(test_dataset["input_ids"], batch, model, device)
        for grp in batch_pred:
            for q_idx, log in grp.items():
                if q_idx not in second_logits:
                    second_logits[q_idx] = log
                else:
                    if AGGREGATION == "MEAN":
                        second_logits[q_idx] += log
                    elif AGGREGATION == "MAX":
                        second_logits[q_idx] = np.maximum(second_logits[q_idx], log)
                    else:  # PRODUCT
                        second_logits[q_idx] *= log

    # Pad CLS/SEP logits for token alignment
    pad = np.zeros((1, len(ID2LABEL)))
    pad[0, 0] = 1e3
    second_logits = {k: np.vstack([pad, v, pad]) for k, v in second_logits.items()}

    # ---------------- Final span predictions -------------------------------
    final_preds: Dict[int, List[str]] = naive_preds.copy()
    for q_idx, log in second_logits.items():
        ents = aggregate_word_predictions(tokens_list[q_idx], log)
        final_preds[q_idx] = [e["label"] for e in ents]

    # ---------------- Evaluation -------------------------------------------
    gold = [[ID2LABEL[t] for t in ex] for ex in test_dataset["ner_tags"]]
    naive = list(naive_preds.values())
    refined = list(final_preds.values())

    print("=" * 100)
    print("Precision / Recall / F1  [Backbone]")
    print(f"{precision_score(gold, naive) * 100:.2f}\t"
          f"{recall_score(gold, naive) * 100:.2f}\t"
          f"{f1_score(gold, naive) * 100:.2f}")
    print("Precision / Recall / F1  [Backbone + CoCoNat]")
    print(f"{precision_score(gold, refined) * 100:.2f}\t"
          f"{recall_score(gold, refined) * 100:.2f}\t"
          f"{f1_score(gold, refined) * 100:.2f}")
    print("# of hard queries:", len(hard_qs), "| # of hard entities:", len(hard_ents))


if __name__ == '__main__':
    main()
