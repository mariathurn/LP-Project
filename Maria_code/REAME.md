# 🧹 T5 Detoxification Models

This repository contains two models for **text detoxification** based on fine-tuning `t5-small`. The goal is to transform toxic input text into more neutral and socially acceptable versions.

## 📂 Contents

- [`Baseline-Model.ipynb`](./Baseline-Model.ipynb):  
  A standard sequence-to-sequence fine-tuning setup using Hugging Face’s `Trainer` on toxic-neutral text pairs.

- [`ModelRefined.ipynb`](./ModelRefined.ipynb):  
  An improved model that adds a **toxicity-aware penalty** using the `unitary/toxic-bert` classifier to discourage toxic outputs during training.

## 🧠 Motivation

While standard fine-tuning improves fluency and semantic accuracy, it doesn't always reduce residual toxicity. The refined model adds a penalty term to the training loss when generated outputs exceed a toxicity threshold — encouraging safer generations.

## 🛠️ Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
