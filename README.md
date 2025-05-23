# CoCoNat
This repository is the official implementation of "CoCoNat: Leveraging Cross-Query Context to Enhance Named Entity Recognition." 


# Abstract
How can we efficiently handle multiple related queries to improve the accuracy and consistency of entity recognition? Named Entity Recognition (NER) is a crucial component in natural language processing, yet accurate and consistent entity recognition across multiple sentences remains a challenge: independent tagging misses cross-sentence cues, whereas large knowledge bases and LLMs are costly to employ.

In this paper, we propose CoCoNat (Context-aware Collective Named-entity Tagging), a lightweight, training-free overlay that enhances pre-trained NER backbones. CoCoNat first isolates a small subset of "hard" queries, significantly reducing the computational cost. A lightweight grouping step then merges only mutually useful queries, which suppresses off-topic noise. Finally, CoCoNat re-applies the backbone within each group and fuses predictions to acheive higher precision and label consistency. Experimental evaluations demonstrate that our collective strategy substantially improves the accuracy of existing NER backbones, yielding up to 9.4%p higher precision and 4.4%p higher F1. Moreover, its model-agnostic design ensures seamless integration with off-the-shelf NER models without retraining, facilitating cost-effective upgrades to the latest architectures.


# Prerequisites
The implementation requires the following libraries.
- datasets==3.4.1  
- evaluate==0.4.3  
- numpy==1.26.4 
- seqeval==1.2.2 
- torch==2.3.0 
- transformers==4.49.0 

# Datasets
The datasets used in our experiments are available at
[CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/), 
[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19), 
[WNUT-2017](https://noisy-text.github.io/2017/emerging-rare-entities.html), 
[FIN](https://aclanthology.org/U15-1010/), 
[BioNLP-2004](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/), 
[BC5CDR](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/),
[MIT Movie](http://groups.csail.mit.edu/sls/), 
[MIT Restaurant](http://groups.csail.mit.edu/sls/).

| Dataset        | Domain      | #Train | #Valid | #Test | #Tag |
| -------------- | ----------- | -----: | ----: | ----: | ---: |
| CoNLL-2003     | News        | 14,041 | 3,250 | 3,453 |    4 |
| OntoNotes 5.0  | Mixed       | 59,924 | 8,528 | 8,262 |   18 |
| WNUT-2017      | Twitter     |  2,395 | 1,009 | 1,287 |    6 |
| FIN            | Finance     |  1,018 |   150 |   305 |    4 |
| BioNLP-2004    | Biomedical  | 16,619 | 1,927 | 3,856 |    5 |
| BC5CDR         | Biomedical  |  5,228 | 5,330 | 5,865 |    2 |
| MIT Movie      | Movie       |  6,816 | 1,000 | 1,953 |   12 |
| MIT Restaurant | FAQ queries |  6,900 |   760 | 1,521 |    8 |

