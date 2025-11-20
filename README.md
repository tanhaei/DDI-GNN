# DDI-LLM: Predicting Unseen Drug–Drug Interactions Using Large Language Models and Molecular Graphs

[![ License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the research paper: **"DDI-LLM: Predicting Unseen Drug–Drug Interactions Using Large Language Models and Molecular Graphs"**.

## 🚀 Overview

**DDI-LLM** is a hybrid deep learning framework designed to predict Drug-Drug Interactions (DDIs), addressing the critical **Cold-Start** problem where one or both drugs are new and have no prior interaction data.

The model integrates two modalities through a novel projection-based Cross-Attention mechanism:
1.  **Structural Modality**: A **Graph Attention Network (GAT)** encodes molecular graphs derived from SMILES strings, incorporating physicochemical descriptors (MW, LogP, TPSA).
2.  **Semantic Modality**: A **frozen Large Language Model (MedGemma)** extracts context-aware embeddings from biomedical texts (e.g., PubMed abstracts).

## ✨ Key Features

* **Strict Cold-Start Protocol**: Evaluation is performed on a test set where **10% of drugs** are completely withheld from training to ensure true generalization.
* **Frozen LLM Strategy**: Utilizes open-source LLMs (MedGemma) in a frozen state to leverage generalized biomedical knowledge.
* **Cross-Attention Fusion**: Implements a specialized attention layer ($\text{Q}_{\text{LLM}}, \text{K}_{\text{GNN}}, \text{V}_{\text{GNN}}$) to align semantic queries with structural keys.

## 🛠️ Installation

### 1. Prerequisites
* Python 3.9 or higher
* CUDA-enabled GPU (**Strongly recommended**), Check Knowledge for GPE operations.

### 2. Clone Repository
```bash
git clone [https://github.com/tanhaei/DDI-GNN.git](https://github.com/tanhaei/DDI-GNN.git)
cd DDI-GNN
```

### 3. Environment SetupIt is recommended to use a virtual environment:Bash# Create environment
python -m venv venv

# Activate Environment:
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

### 4. Install Dependencies (Critical Step)
Note: Graph Neural Network libraries (DGL) and PyTorch must match your CUDA version.

Step A: Install PyTorch (Example for CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```
(Check pytorch.org for your specific CUDA version command):

Step B: Install DGL (Deep Graph Library)

```bash
pip install dgl -f [https://data.dgl.ai/wheels/cu118/repo.html](https://data.dgl.ai/wheels/cu118/repo.html)
```

(If you do not have a GPU, run pip install dgl):

Step C: Install Remaining Requirements

```bash
pip install -r requirements.txt
```
### 5. HuggingFace Authentication (For MedGemma)
The model uses a variant of Gemma, which may require authentication.

Accept the model terms on HuggingFace.

Log in via terminal:

```bash
huggingface-cli login
# Paste your Access Token
```

### 🏃 Usage
To run the complete pipeline (Preprocessing → Training ... → Cold-Start Evaluation):

```bash
python main.py
```

### Pipeline Execution Summary
Graph/Feature Generation: SMILES are converted to molecular graphs, and molecular descriptors are calculated.

Data Splitting: The pipeline executes the strict Cold-Start split, ensuring 10% of drugs are unseen in the test set.

LLM Pre-computation: Text embeddings are extracted using the frozen MedGemma model and cached in memory for fast training.

Training: The GNN and Fusion head are trained for 50 epochs (default).

Evaluation: Final metrics (F1, AUC, Precision, Recall) are reported on the unseen Cold-Start test set.

### 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
