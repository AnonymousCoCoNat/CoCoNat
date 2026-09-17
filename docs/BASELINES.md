# Recent-baseline integration

The baseline layer separates **execution** from **evaluation**. Official systems may run
in their own environments, but every result must cross the same strict word-offset
boundary before the common evaluator scores it.

ScdNER, PromptNER, GPT-NER, IRRA, and ReasoningNER are not interchangeable prompts.
Their papers define different supervision, retrieval, and reasoning procedures. This
repository therefore does not imitate those names with one generic LLM call. Use the
authors' implementation/checkpoint and either invoke it through `external_command` or
import its verified predictions through `external_predictions`.

The generic `local_llm` and `hosted_llm` adapters are appropriate for the paper's common
instruction-LLM comparison. Their output is labelled as the configured checkpoint plus
`generic_indexed_word_json_v1`, never as PromptNER or ReasoningNER.

Relevant primary sources:

- GLiNER official repository: <https://github.com/urchade/GLiNER>
- PromptNER paper: <https://aclanthology.org/2023.acl-long.698/>
- ScdNER paper: <https://aclanthology.org/2023.emnlp-main.970/>
- IRRA official repository: <https://github.com/DMIRLAB-Group/IRRA>
- IRRA paper: <https://aclanthology.org/2025.naacl-long.540/>
- ReasoningNER official repository: <https://github.com/HuiResearch/ReasoningIE>
- Hugging Face token-classification task guide:
  <https://huggingface.co/docs/transformers/v4.49.0/en/tasks/token_classification>

## Common prediction format

The normalized JSONL file contains exactly one row per exported query:

```json
{"id":"q1","tokens":["Acme","opened","."],"spans":[{"start":0,"end":1,"label":"ORG","score":0.94}],"metadata":{}}
```

Requirements:

- IDs are unique and cover the exported collection exactly.
- `tokens` exactly equals the input token array.
- `start` and `end` are integer, zero-based, half-open word offsets.
- every type is covered by the configured `label_map`;
- `score`, when present, is a finite value in `[0,1]`;
- duplicate or out-of-bounds spans are rejected.

If an official system emits character offsets, run `convert-predictions --format char`.
Only exact boundaries in the canonical single-space rendering are accepted. If it emits
BIO tags, use `--format bio`. If it emits free text, use `--format generation`; invalid
items are counted as misses and are never corrected with gold labels.

## Source manifest

Every external integration requires JSON containing concrete values:

```json
{
  "method": "ReasoningNER",
  "source_url": "https://github.com/OWNER/REPOSITORY",
  "source_revision": "full commit SHA or immutable release tag",
  "checkpoint": "exact checkpoint identifier and revision",
  "supervision": "released zero-shot setting",
  "notes": "dataset-specific flags, prompt file hash, or retrieval index identity"
}
```

Copy `configs/source_manifest.example.json` and replace every placeholder. Unedited
templates are rejected. Store the manifest next to the corresponding predictions or
adapter configuration.

## Prediction import

Use this path when an official program has already produced output:

```yaml
- name: ReasoningNER
  kind: external_predictions
  family: reasoning NER
  regime: zero-shot
  enabled: true
  prediction_path: ../external/reasoningner/conll2003.predictions.jsonl
  source_manifest: ../external/reasoningner/source.json
  label_map:
    PER: PER
    ORG: ORG
    LOC: LOC
    MISC: MISC
```

## External command

Use this path when a wrapper around the official code can accept the bridge request:

```yaml
- name: ScdNER
  kind: external_command
  family: document context
  regime: full supervision
  enabled: true
  cwd: ../third_party/scdner
  command:
    - python
    - coconat_adapter.py
    - --input
    - "{input}"
    - --output
    - "{output}"
  source_manifest: ../external/scdner/source.json
```

The bridge writes one temporary JSON request:

```json
{
  "protocol_version": 1,
  "labels": ["PER", "ORG", "LOC", "MISC"],
  "queries": [{"id":"q1","tokens":["Acme","opened","."],"text":"Acme opened .","doc_id":"d1"}]
}
```

There is no gold field. The command must write normalized prediction JSONL to `{output}`.
It is invoked as an argument list with `shell=False`; placeholders may occur anywhere in
an argument. On failure, run the configured command directly to inspect the official
program's own stderr.

External-command timing includes interpreter/process startup and model loading on every
measured invocation. Report that scope. For a resident-server or in-process comparison,
write a native `Backend` adapter so that loading occurs once outside the timed region.

## Native adapter configuration

Hugging Face token classifier:

```yaml
- name: Longformer-base-4096
  kind: hf
  family: long context
  regime: full supervision
  enabled: true
  checkpoint: ../../checkpoints/conll2003/longformer-base-4096
  max_length: 4096
  batch_size: 2
  device: auto
  context_mode: document
```

GLiNER:

```yaml
- name: GLiNER-large
  kind: gliner
  family: generalist encoder
  regime: zero-shot
  enabled: true
  checkpoint: urchade/gliner_large-v2.1
  threshold: 0.5
  device: auto
  label_map:
    person: PER
    organization: ORG
    location: LOC
    miscellaneous: MISC
```

The left side of `label_map` is the model-facing semantic name; the right side is the
dataset type. GLiNER's map must cover the dataset types exactly. Record the checkpoint
revision resolved during the final run.

Local instruction model:

```yaml
- name: Llama-3.1-8B-Instruct-common-prompt
  kind: local_llm
  family: instruction LLM
  regime: five-shot per type
  enabled: true
  checkpoint: meta-llama/Meta-Llama-3.1-8B-Instruct
  shots_per_type: 5
  seed: 42
  device: auto
  max_length: 8192
  max_new_tokens: 1024
```

The demonstration selector draws distinct training sentences independently for each
entity type, then deduplicates their union.

Hosted instruction model:

```yaml
- name: GPT-4o-mini-2024-07-18-common-prompt
  kind: hosted_llm
  family: instruction LLM
  regime: five-shot per type
  enabled: true
  checkpoint: gpt-4o-mini-2024-07-18
  shots_per_type: 5
  seed: 42
  api_key_env: OPENAI_API_KEY
```

The command requires `--allow-api`. It uses deterministic temperature-zero Chat
Completions, requests JSON output, stores response metadata, and does not retry an
authentication/permission failure.

