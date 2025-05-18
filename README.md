# CoCoNat
This repository is the official implementation of "CoCoNat: Leveraging Cross-Query Context to Enhance Named Entity Recognition." 


# Abstract
How can we efficiently handle multiple related queries to improve the accuracy and consistency of entity recognition? Named Entity Recognition (NER) is a crucial component in natural language processing, yet accurate and consistent entity recognition across multiple sentences remains a challenge: independent tagging misses cross-sentence cues, whereas large knowledge bases and LLMs are costly to employ.

In this paper, we propose CoCoNat (Context-aware Collective Named-entity Tagging), a lightweight, training-free overlay that enhances pre-trained NER backbones. CoCoNat first isolates a small subset of "hard" queries, significantly reducing the computational cost. A lightweight grouping step then merges only mutually useful queries, which suppresses off-topic noise. Finally, CoCoNat re-applies the backbone within each group and fuses predictions to acheive higher precision and label consistency. Experimental evaluations demonstrate that our collective strategy substantially improves the accuracy of existing NER backbones, yielding up to 9.4%p higher precision and 4.4%p higher F1. Moreover, its model-agnostic design ensures seamless integration with off-the-shelf NER models without retraining, facilitating cost-effective upgrades to the latest architectures.


# Prerequisites
The implementation requires the following libraries.

...

# Datasets
...
