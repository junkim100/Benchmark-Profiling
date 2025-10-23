# EMNLP 2025: Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2510.01232-b31b1b.svg)](https://arxiv.org/abs/2510.01232)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code for the EMNLP 2025 main paper "Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks". Our work introduces a novel approach to understanding how large language models (LLMs) process different types of reasoning tasks by analyzing the internal mechanisms that drive benchmark performance.

## 🎯 Overview

Benchmark Profiling is a mechanistic interpretability method that identifies and analyzes the specific neural network regions responsible for different cognitive abilities in LLMs. By selectively damaging these regions, we can:

- **Identify critical parameters** for specific reasoning abilities
- **Understand cross-benchmark relationships** and shared cognitive mechanisms
- **Provide mechanistic insights** into how LLMs solve different types of problems
- **Enable targeted model analysis** for specific cognitive capabilities

## 🏗️ Project Structure

```text

├── damage_region/           # Model damage experiments
│   └── damage_model.py      # Apply damage to critical regions
├── data_preprocess/         # Dataset preprocessing pipeline
│   ├── download_dataset.py  # Download datasets from HuggingFace
│   ├── preprocess.py        # Data preprocessing utilities
│   └── transform.py         # Data transformation functions
├── extract_region/          # Parameter region extraction
│   └── extract_region.py    # Extract critical parameters and save selections
├── training/                # Model fine-tuning components
│   ├── step1_supervised_finetuning/ # SFT training scripts
│   └── utils/              # Training utilities

├── config.yml              # Main configuration file
├── run.sh                  # Main execution script
├── requirements.txt         # Python dependencies
├── CITATION.bib            # Academic citation
├── CONTRIBUTING.md         # Contribution guidelines
└── LICENSE                 # Apache License 2.0
```

**Note**: The repository includes only the core code. Datasets, experimental results, and generated figures are excluded and should be downloaded/generated separately.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s)
- PyTorch 2.0+
- Transformers library
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/junkim100/Unveiling-Regions.git
cd Unveiling-Regions
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Download datasets:

```bash
# Use the provided script to download required datasets
python data_preprocess/download_dataset.py
```

1. Configure your setup:

```bash
# Edit config.yml to specify your models, datasets, and hardware configuration
vim config.yml
```

**Note**: This repository contains only the core code. Datasets and experimental outputs are not included and must be downloaded/generated separately.

### Basic Usage

Run the complete benchmark profiling pipeline:

```bash
# Run full pipeline (training, extraction, damage, evaluation)
./run.sh

# Run evaluation only (skip training and extraction)
./run.sh -e
```

### Advanced Usage Examples

#### 1. Extract Critical Regions

```bash
# Extract regions for specific model and dataset
cd extract_region
python extract_region.py generate_masks \
    --input_dir ./outputs/Analogical_Reasoning/llama3.1/train/checkpoint_full \
    --output_dir ./outputs/Analogical_Reasoning/llama3.1/extract/checkpoint_full \
    --k 0.01024
```

#### 2. Apply Damage to Models

```bash
# Damage model using extracted regions
cd damage_region
python damage_model.py \
    ./outputs/Analogical_Reasoning/llama3.1/extract \
    ./outputs/Analogical_Reasoning/llama3.1/damage \
    meta-llama/Llama-3.1-8B-Instruct \
    0.01024
```
#### 4. Evaluate Models

```bash
# Evaluate damaged model on specific benchmarks
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=./outputs/Analogical_Reasoning/llama3.1/damage/checkpoint_full/top0.01024 \
    --tasks gsm8k,arc_challenge,hellaswag \
    --batch_size 8
```

## 📊 Supported Benchmarks

Our framework supports analysis across multiple cognitive reasoning domains:

- **Analogical Reasoning** - Pattern recognition and analogy completion
- **Commonsense & Causal Reasoning** - Common sense understanding and causal relationships
- **Contextual Recall** - Information retrieval from context
- **Deductive Reasoning** - Logical deduction and inference
- **Inductive Reasoning** - Pattern generalization and rule learning
- **Long-term Knowledge** - Factual knowledge retrieval
- **Quantitative Reasoning** - Mathematical and numerical reasoning
- **Semantic Relationship** - Understanding semantic connections
- **Spatial Reasoning** - Spatial relationship understanding
- **Temporal Reasoning** - Time-based logical reasoning

## 🔧 Configuration

The main configuration is handled through `config.yml`:

```yaml
settings:
  cuda_visible_devices: 0,1,2,3,4,5,6,7
  k_values: [0.00001, 0.00004, 0.00016, 0.00064, 0.00256, 0.01024]

models:
  - name: meta-llama/Llama-3.1-8B-Instruct
    tokenizer: llama3.1

evals:
  benchmarks: ["Inductive_Reasoning", "Analogical_Reasoning"]
  num_fewshot: [0, 0]
```
## 🔬 Methodology

Our approach consists of four main stages:

1. **Fine-tuning**: Adapt models to specific reasoning tasks
2. **Region Extraction**: Identify critical parameters using gradient-based methods
3. **Selective Modification**: Apply targeted damage
4. **Evaluation**: Assess performance changes across benchmarks

## 📚 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{kim2025benchmark,
  title={Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks},
  author={Kim, Dongjun and Shim, Gyuho and Chun, Yongchan and Kim, Minhyuk and Park, Chanjun and Lim, Heuiseok},
  journal={arXiv preprint arXiv:2510.01232},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors.

---

**Note**: This repository is actively maintained and updated. Please check for the latest version and updates.
