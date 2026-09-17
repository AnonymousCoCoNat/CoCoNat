"""Lazy loading keeps offline analysis independent of GPU and hosted API packages."""


def build_backend(config, labels, train=(), allow_api=False):
    config = dict(config)
    kind = config.pop("kind")
    config.pop("enabled", None)
    config.pop("family", None)
    config.pop("regime", None)
    # Metadata name is passed only where needed; model identity otherwise is its checkpoint.
    if kind == "toy":
        from .toy import ToyBackend
        return ToyBackend(labels=labels, **config)
    if kind == "hf":
        from .hf import HFBackend
        return HFBackend(labels=labels, **config)
    if kind in {"gliner", "flair", "spacy"}:
        from .native import FlairBackend, GLiNERBackend, SpacyBackend
        cls = {"gliner": GLiNERBackend, "flair": FlairBackend, "spacy": SpacyBackend}[kind]
        return cls(labels=labels, **config)
    if kind in {"external_command", "external_predictions"}:
        from .external import ExternalCommand, ExternalPredictions
        cls = ExternalCommand if kind == "external_command" else ExternalPredictions
        return cls(labels=labels, **config)
    if kind in {"local_llm", "hosted_llm"}:
        from .llm import HostedLLMBackend, LocalLLMBackend, select_demonstrations
        shots = config.pop("shots_per_type", 5)
        demonstrations = select_demonstrations(train, labels, shots, config.pop("seed", 42))
        if kind == "hosted_llm":
            config["allow_api"] = allow_api
        cls = HostedLLMBackend if kind == "hosted_llm" else LocalLLMBackend
        return cls(labels=labels, demonstrations=demonstrations, **config)
    raise ValueError(f"Unknown backend kind: {kind}")
