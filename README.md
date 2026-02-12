# CoCoNat
This repository is the official implementation of "CoCoNat: Leveraging Cross-Query Context to Enhance Named Entity Recognition." 


## Abstract
How can we efficiently handle multiple related queries to improve the accuracy and consistency of entity recognition? Named Entity Recognition (NER) is a crucial component in natural language processing, yet accurate and consistent entity recognition across multiple sentences remains a challenge: independent tagging misses cross-sentence cues, whereas large knowledge bases and LLMs are costly to employ.

In this paper, we propose CoCoNat (Context-aware Collective Named-entity Tagging), a lightweight, training-free overlay that enhances pre-trained NER backbones. CoCoNat first isolates a small subset of "hard" queries, significantly reducing the computational cost. A lightweight grouping step then merges only mutually useful queries, which suppresses off-topic noise. Finally, CoCoNat re-applies the backbone within each group and fuses predictions to acheive higher precision and label consistency. Experimental evaluations demonstrate that our collective strategy substantially improves the accuracy of existing NER backbones, yielding up to 9.4%p higher precision and 4.4%p higher F1. Moreover, its model-agnostic design ensures seamless integration with off-the-shelf NER models without retraining, facilitating cost-effective upgrades to the latest architectures.


## Prerequisites
The implementation requires the following libraries.
- datasets==3.4.1  
- evaluate==0.4.3  
- numpy==1.26.4 
- seqeval==1.2.2 
- torch==2.3.0
- tqdm==4.67.1
- transformers==4.49.0

## Quick start
The script `main.py` can (1) download data, (2) fine-tune a backbone NER model, and (3) run evaluation with/without the CoCoNat refinement.

```bash
# Runs training + evaluation on CoNLL-2003 with DeBERTa-v3-base (defaults in main.py)
python main.py
```

On first run, the dataset will be **auto-downloaded** via Hugging Face Datasets and cached under `./output/conll2003`.

## Fine-tuning

Open `main.py` and check the **Model & dataset paths** block:

```python
TRAIN = True
DATASET_NAME = "conll2003"
DATASET_LOC  = "conll2003"

MODEL_NAME = "DeBERTa-v3-base"
MODEL_CHECKPOINT = "microsoft/deberta-v3-base"

LOCAL_FOLDER = f"./output/{DATASET_NAME}"
LOCAL_FOLDER_MODEL = f"{LOCAL_FOLDER}/{MODEL_NAME}"
```

Then run:

```bash
python main.py
```

Artifacts (checkpoints, logs) will be saved under:

```
./output/<DATASET_NAME>/<MODEL_NAME>/
```

To adjust training settings (epochs, LR, batch size, etc.), edit the `TrainingArguments` in `main.py`.


## Evaluation

1. Set a valid checkpoint path:

```python
LOCAL_CKPT = "./output/conll2003/DeBERTa-v3-base/checkpoint-5000"
```

2. Switch to inference mode:

```python
TRAIN = False
```

3. Run:

```bash
python main.py
```

At the end, the script prints **Precision / Recall / F1** for:

* the backbone alone, and
* **backbone + CoCoNat** (after second-pass grouping/aggregation).



## Tips & Troubleshooting

* **GPU usage**: Transformers/Trainer will use GPU automatically if available.
* **OOM issues**: Lower `per_device_train_batch_size` or `model_max_length`.
* **Different model**: Change `MODEL_CHECKPOINT` (e.g., `bert-base-cased`, `xlm-roberta-base`).






