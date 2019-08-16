# Multitask Learning for Biomedical Named Entity Recognition with Cross-Sharing Structure
This repository is a Biomedical Named Entity Recognition model. The code is based on [XuezheMax/NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2).

## Requirements

Python 3+, PyTorch < 1.0, Gensim >= 0.12.0

## Running the experiments

In the root of the repository, first make the tmp directory:

    mkdir tmp

To train a Baseline Single-task Model (STM) model,

    ./run_ner_crf.sh

To train a Fully-shared Multi-task Model (FS-MTM) model,

    ./run_fullyshare.sh

To train a Shared-private Multi-task Model (SP-MTM) model, (Specify adv\_loss\_coef = 0 and diff\_loss\_coef = 0 in config file.)

    ./run_adversarial.sh 

To train a Adversarial Multi-task Model (ADV-MTM) model,

    ./run_adversarial.sh

To train a Cross-sharing Multi-task Model (CS-MTM) model,

    ./run_crossshare.sh

To make a grid search,

    python3 grid_search.py
