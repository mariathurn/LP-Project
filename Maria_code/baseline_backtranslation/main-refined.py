import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast,
)
from sentence_transformers import SentenceTransformer
import numpy as np

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "output_data")

os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)


class BackTranslationBaseline:
    def __init__(
        self,
        device: str = "cuda",
        translation_model_name: str = "facebook/nllb-200-distilled-600M",  # Use smaller model
        detox_model_name: str = "s-nlp/bart-base-detox",
        similarity_threshold: float = 0.75
    ):
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        self.lang_id_mapping = {
            "ru": "rus_Cyrl",
            "en": "eng_Latn",
            "am": "amh_Ethi",
            "es": "spa_Latn",
            "uk": "ukr_Cyrl",
            "zh": "zho_Hans",
            "ar": "arb_Arab",
            "hi": "hin_Deva",
            "de": "deu_Latn",
            "tt": "tat_Cyrl",
            "fr": "fra_Latn",
            "it": "ita_Latn",
            "he": "heb_Hebr",
            "ja": "jpn_Jpan",
        }

        print(f"Loading translation model ({translation_model_name})...")
        self.translation_tokenizer = AutoTokenizer.from_pretrained(
            translation_model_name)
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            translation_model_name)
        self.translation_model = self.translation_model.eval().to(self.device)

        print(f"Loading detoxification model ({detox_model_name})...")
        self.detox_model = BartForConditionalGeneration.from_pretrained(
            detox_model_name)
        self.detox_model = self.detox_model.eval().to(self.device)
        self.detox_tokenizer = BartTokenizerFast.from_pretrained(
            detox_model_name)

        # Add similarity model
        print("Loading sentence similarity model...")
        self.similarity_model = SentenceTransformer(
            'all-MiniLM-L6-v2').to(self.device)
        self.similarity_threshold = similarity_threshold

    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str, batch_size: int = 32, max_length: int = 128, verbose: bool = True) -> List[str]:
        self.translation_tokenizer.src_lang = self.lang_id_mapping[src_lang]
        self.translation_tokenizer.tgt_lang = self.lang_id_mapping[tgt_lang]

        translations = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(
                iterator, desc=f"Translating {src_lang}→{tgt_lang}")

        for i in iterator:
            batch = texts[i: i + batch_size]
            tokenized = self.translation_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            outputs = self.translation_model.generate(
                **tokenized,
                forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[
                    self.translation_tokenizer.tgt_lang],
                max_new_tokens=max_length
            )
            translations.extend(self.translation_tokenizer.batch_decode(
                outputs, skip_special_tokens=True))
        return translations

    def detoxify_batch(self, texts: List[str], batch_size: int = 32, verbose: bool = True) -> List[str]:
        detoxified = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Detoxifying")

        for i in iterator:
            batch = texts[i: i + batch_size]
            tokenized = self.detox_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.detox_model.generate(
                **tokenized,
                do_sample=True,
                top_p=0.9,  # Nucleus sampling
                temperature=0.8,
                max_new_tokens=128
            )
            detoxified.extend(self.detox_tokenizer.batch_decode(
                outputs, skip_special_tokens=True))
        return detoxified

    def compute_similarity(self, source_texts: List[str], target_texts: List[str]) -> List[float]:
        """Compute semantic similarity between source and target sentences."""
        source_embeddings = self.similarity_model.encode(
            source_texts, convert_to_tensor=True)
        target_embeddings = self.similarity_model.encode(
            target_texts, convert_to_tensor=True)

        # Compute cosine similarity
        similarities = []
        for i in range(len(source_texts)):
            similarity = torch.nn.functional.cosine_similarity(
                source_embeddings[i].unsqueeze(0),
                target_embeddings[i].unsqueeze(0)
            ).item()
            similarities.append(similarity)

        return similarities

    def filter_detoxified_texts(self, original_texts: List[str], detoxified_texts: List[str],
                                save_dropped_path: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """Filter detoxified texts based on similarity threshold."""
        similarities = self.compute_similarity(
            original_texts, detoxified_texts)

        filtered_texts = []
        kept_indices = []
        dropped = {"original": [], "detoxified": [], "similarity": []}

        for i, (orig, detox, sim) in enumerate(zip(original_texts, detoxified_texts, similarities)):
            if sim >= self.similarity_threshold:
                filtered_texts.append(detox)
                kept_indices.append(i)
            else:
                # Keep original text if similarity is too low
                filtered_texts.append(orig)
                dropped["original"].append(orig)
                dropped["detoxified"].append(detox)
                dropped["similarity"].append(sim)

        # Save dropped examples if requested
        if save_dropped_path and dropped["original"]:
            with open(save_dropped_path, 'w', encoding='utf-8') as f:
                json.dump(dropped, f, ensure_ascii=False, indent=2)

        return filtered_texts, kept_indices

    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 64,
        verbose: bool = True,
        save_dropped_path: Optional[str] = None
    ):
        print(f"Loading data from {input_path}...")
        data = pd.read_csv(input_path, sep="\t")
        output_path = Path(output_path)

        translations = {lang: [] for lang in self.lang_id_mapping}
        for lang in tqdm(self.lang_id_mapping, desc="Translating to English", disable=not verbose):
            lang_data = data[data.lang == lang].toxic_sentence.tolist()
            if not lang_data:
                continue
            if lang == "en":
                translations[lang] = lang_data
            else:
                translations[lang] = self.translate_batch(
                    lang_data, src_lang=lang, tgt_lang="en", batch_size=batch_size, verbose=verbose)

        detoxified = {lang: [] for lang in translations}
        for lang in tqdm(translations, desc="Detoxifying", disable=not verbose):
            if lang == "ru":
                continue
            if translations[lang]:
                detoxified[lang] = self.detoxify_batch(
                    translations[lang], batch_size=batch_size, verbose=verbose)

        filtered_detoxified = {lang: [] for lang in detoxified}
        for lang in tqdm(detoxified, desc="Filtering by similarity", disable=not verbose):
            if not detoxified.get(lang) or lang == "ru":
                continue

            # Get original texts for comparison
            original_texts = translations[lang]
            detox_texts = detoxified[lang]

            # Apply similarity filtering
            filtered_texts, _ = self.filter_detoxified_texts(
                original_texts,
                detox_texts,
                save_dropped_path=save_dropped_path if lang == "en" else None
            )

            filtered_detoxified[lang] = filtered_texts

        backtranslations = {lang: [] for lang in self.lang_id_mapping}
        for lang in tqdm(self.lang_id_mapping, desc="Backtranslating", disable=not verbose):
            if not filtered_detoxified.get(lang):
                continue
            if lang == "en":
                backtranslations[lang] = filtered_detoxified[lang]
            else:
                backtranslations[lang] = self.translate_batch(
                    filtered_detoxified[lang], src_lang="en", tgt_lang=lang, batch_size=batch_size, verbose=verbose)

        print(f"Saving results to {output_path}...")
        os.makedirs(output_path.parent, exist_ok=True)
        lang_indices = {lang: 0 for lang in backtranslations}
        neutral_sentences = []

        for _, row in data.iterrows():
            lang = row["lang"]
            detox_list = backtranslations.get(lang, [])
            idx = lang_indices.get(lang, 0)
            if idx < len(detox_list):
                neutral_sentences.append(detox_list[idx])
                lang_indices[lang] += 1
            else:
                neutral_sentences.append("")
        data["neutral_sentence"] = neutral_sentences
        data.to_csv(output_path, sep="\t", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtranslation-based text detoxification pipeline")
    parser.add_argument("--input_path", type=str,
                        default=Path(INPUT_DATA_PATH, "dev_inputs.tsv"))
    parser.add_argument("--output_path", type=str,
                        default=Path(OUTPUT_DATA_PATH, "baseline_backtranslation_dev.tsv"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dropped_path", type=str, default=None,
                        help="Path to save examples that didn't meet similarity threshold")
    parser.add_argument("--similarity_threshold", type=float, default=0.75,
                        help="Threshold for semantic similarity filtering")
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Initializing backtranslation pipeline...")
    pipeline = BackTranslationBaseline(
        device=args.device,
        translation_model_name="facebook/nllb-200-distilled-600M",  # Use smaller model
        similarity_threshold=args.similarity_threshold
    )
    print("Processing dataset...")
    pipeline.process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        verbose=args.verbose,
        save_dropped_path=args.save_dropped_path
    )
