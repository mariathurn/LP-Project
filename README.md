Multilingual Text Detoxification

This repository contains code, models, and evaluation scripts for a project on multilingual text detoxification, focusing on English and German. The aim is to reduce toxicity in text while preserving semantic content and fluency.

📌 Overview

We compare two detoxification approaches:
	•	Delete-Only Baseline: A rule-based method that removes predefined toxic spans from the input.
	•	Refined Model: A fine-tuned t5-small model trained on English toxic/non-toxic pairs from the s-nlp/paradetox dataset.

German detoxification is evaluated using a translation-based pipeline: German → English → detoxify → German.

📁 Project Structure

├── main.py                          # Delete-only baseline
├── Model-refined.ipynb              # Translation-based detoxification with refined T5 model
├── xcoment_fluency.ipynb            # Fluency evaluation using XCOMET-lite
├── input_data/                      # Datasets in JSON format
├── output_data/                     # Detoxified outputs for both EN and DE
├── evaluation/                      # Evaluation scripts and logs
└── requirements.txt                 # Required Python packages

🧪 Evaluation Metrics

The following metrics are used:
	•	Toxicity Score: XLM-Roberta classifier for binary toxicity classification.
	•	Content Preservation: Cosine similarity between LaBSE sentence embeddings.
	•	Fluency: COMET-style XCOMETLite model estimates output fluency without references.

📊 Results Summary

Method	EN Non-Toxic (%)	DE Non-Toxic (%)	EN LaBSE	DE LaBSE	EN Fluency	DE Fluency
Delete-Only Baseline	71.25	39.00	0.9478	0.9810	0.6290	0.9424
Refined Model	68.00	56.25	0.9089	0.8848	0.8715	0.8999

🚀 How to Run
	1.	Install dependencies

pip install -r requirements.txt


	2.	Run Delete Baseline

python main.py


	3.	Run Refined Model Detoxification
Open and run Model-refined.ipynb to apply the translation-based pipeline.
	4.	Evaluate Fluency
Run xcoment_fluency.ipynb to compute XCOMETLite scores.

📖 Citation

If you use this codebase or dataset in your research, please cite relevant references such as:
	•	Dementieva et al. (2023). Exploring methods for cross-lingual text style transfer.
	•	Sushko (2024). Synthetic training for multilingual detoxification.
	•	Luo et al. (2024). Translation-based detox pipelines with post-processing.
