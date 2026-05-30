# LLM Priming Effects Research

This project investigates the impact of model scale, syntactic complexity, and fine-tuning strategies on priming-induced biases in large language models.

## Project Overview

Our research explores how different factors affect an LLM's susceptibility to priming effects:

1. **Model Scale**: How does increasing model size (e.g., from 1.5B to 14B parameters) affect priming susceptibility?
2. **Syntactic Complexity**: Do more complex syntactic structures (e.g., nested clauses, long-distance dependencies) amplify priming effects?
3. **Fine-tuning Strategies**: Can techniques like SFT and RLHF mitigate priming-induced biases?
```
llm_priming_project/
├── data/
│   ├── raw/               # Original datasets
│   │   ├── cola/          # CoLA (Corpus of Linguistic Acceptability)
│   │   └── blimp/         # BLiMP (Benchmark of Linguistic Minimal Pairs)
│   └── processed/         # Processed datasets for experiments
├── models/                # Model checkpoints and configurations
├── scripts/
│   ├── data_processing/   # Scripts for data preparation
│   ├── evaluation/        # Model evaluation scripts
│   ├── training/          # Fine-tuning scripts
│   └── utils/             # Utility functions
├── results/               # Experimental results
└── logs/                  # Log files
```
## Current Progress

- [x] Project setup and environment configuration
- [x] Data acquisition
  - [x] Downloaded CoLA dataset
  - [x] Downloaded BLiMP dataset (key phenomena)
- [ ] Data processing
- [ ] Baseline evaluation
- [ ] Fine-tuning
- [ ] Analysis and results

## Datasets

### CoLA (Corpus of Linguistic Acceptability)
A collection of English sentences annotated with grammatical acceptability judgments. The dataset helps establish a baseline for grammatical assessment capabilities.

### BLiMP (Benchmark of Linguistic Minimal Pairs)
A challenge set for evaluating what language models know about specific grammatical phenomena. Each example consists of a minimal pair: two sentences that differ in only one respect and showcase a specific linguistic phenomenon.

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/YichaoYangThomas/llm_priming_project.git
cd llm_priming_project

# Create conda environment (preferably in /scratch to avoid quota issues)
conda create --prefix /scratch/your_id/conda_envs/llm_priming python=3.9
conda activate /scratch/your_id/conda_envs/llm_priming

# Install dependencies
pip install -r requirements.txt

# Download datasets
python scripts/data_processing/download_datasets.py
