# CoCoNat
This repository is the official implementation of "CoCoNat: Leveraging Cross-Query Context to Enhance Named Entity Recognition." 


# Abstract
How can we efficiently handle multiple related queries to improve the accuracy and consistency of entity recognition?
Named Entity Recognition (NER) is a crucial component in natural language processing, 
yet accurate and consistent entity recognition across multiple sentences remains a challenge:
independent tagging misses cross-sentence cues, whereas large knowledge bases and LLMs are costly to employ.

In this paper, we propose CoCoNat (Context-aware Collective Named-entity Tagging), a lightweight, training-free overlay that enhances any pre-trained NER backbone.
CoCoNat first flags low-confidence or label-conflicting entities, thereby isolating a small subset of "hard" queries.
It then forms context-coherent groups by linking queries through exact entity overlap or embedding similarity,
and re-applies the same tagger to each group after a confidence-aware, gradual concatenation.
Experimental evaluations demonstrate that this collective strategy substantially enhances both accuracy and consistency in entity recognition.
Moreover, its model-agnostic design ensures seamless integration with any off-the-shelf NER model without retraining, facilitating cost-effective upgrades to the latest architectures.

# Prerequisites
The implementation requires the following libraries.

...

# Datasets
...
