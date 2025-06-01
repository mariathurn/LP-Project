import argparse
import os
from pathlib import Path
from typing import List
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast,
)
import csv
import json
from sentence_transformers import SentenceTransformer, util

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "output_data")

os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)


def convert_tsv_to_json(tsv_path: str, json_path: str):
    data = []
    with open(tsv_path, "r", encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        for row in reader:
            entry = {
                "input_text": row.get("toxic_sentence", ""),
                "target_text": row.get("neutral_sentence", "")
            }
            data.append(entry)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)

    print(f"Converted {len(data)} entries to JSON format.")


class BackTranslationBaseline:
    def __init__(
        self,
        device: str = "cuda",
        translation_model_name: str = "facebook/nllb-200-3.3B",
        detox_model_name: str = "s-nlp/bart-base-detox",
        similarity_threshold: float = 0.7,
        save_dropped_path: str = None
    ):
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.similarity_threshold = similarity_threshold
        self.save_dropped_path = save_dropped_path
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dropped = []

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
        self.translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
        self.translation_model = self.translation_model.eval().to(self.device)

        print(f"Loading detoxification model ({detox_model_name})...")
        self.detox_model = BartForConditionalGeneration.from_pretrained(detox_model_name)
        self.detox_model = self.detox_model.eval().to(self.device)
        self.detox_tokenizer = BartTokenizerFast.from_pretrained(detox_model_name)

    def is_similar(self, a: str, b: str) -> bool:
        emb1 = self.sim_model.encode(a, convert_to_tensor=True)
        emb2 = self.sim_model.encode(b, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item() > self.similarity_threshold

    def filter_by_similarity(self, inputs: List[str], outputs: List[str]) -> List[str]:
        filtered = []
        for inp, out in zip(inputs, outputs):
            if self.is_similar(inp, out):
                filtered.append(out)
            else:
                filtered.append("")
                if self.save_dropped_path is not None:
                    self.dropped.append({"input": inp, "output": out})
        return filtered

    def save_dropped(self):
        if self.save_dropped_path and self.dropped:
            with open(self.save_dropped_path, "w", encoding="utf-8") as f:
                json.dump(self.dropped, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.dropped)} dropped examples to {self.save_dropped_path}")

    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str, batch_size: int = 32, max_length: int = 128, verbose: bool = True) -> List[str]:
        self.translation_tokenizer.src_lang = self.lang_id_mapping[src_lang]
        self.translation_tokenizer.tgt_lang = self.lang_id_mapping[tgt_lang]

        translations = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc=f"Translating {src_lang}→{tgt_lang}")

        for i in iterator:
            batch = texts[i: i + batch_size]
            tokenized = self.translation_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            outputs = self.translation_model.generate(
                **tokenized,
                forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[
                    self.translation_tokenizer.tgt_lang],
                max_new_tokens=max_length,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                num_beams=4,
                early_stopping=True,
            )
            translations.extend(self.translation_tokenizer.batch_decode(outputs, skip_special_tokens=True))
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
                max_new_tokens=128,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                num_beams=4,
                early_stopping=True,
            )
            detoxified.extend(self.detox_tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return detoxified

    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 64,
        verbose: bool = True
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
                raw = self.detoxify_batch(translations[lang], batch_size=batch_size, verbose=verbose)
                detoxified[lang] = self.filter_by_similarity(translations[lang], raw)

        backtranslations = {lang: [] for lang in self.lang_id_mapping}
        for lang in tqdm(self.lang_id_mapping, desc="Backtranslating", disable=not verbose):
            if not detoxified.get(lang):
                continue
            if lang == "en":
                backtranslations[lang] = detoxified[lang]
            else:
                raw_bt = self.translate_batch(
                    detoxified[lang], src_lang="en", tgt_lang=lang, batch_size=batch_size, verbose=verbose)
                backtranslations[lang] = self.filter_by_similarity(translations[lang], raw_bt)

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

        self.save_dropped()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtranslation-based text detoxification pipeline")
    parser.add_argument("--input_path", type=str,
                        default=Path(INPUT_DATA_PATH, "dev_inputs.tsv"))
    parser.add_argument("--output_path", type=str,
                        default=Path(OUTPUT_DATA_PATH, "baseline_backtranslation_dev.tsv"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--similarity_threshold", type=float, default=0.7)
    parser.add_argument("--save_dropped_path", type=str, default=None)
    parser.add_argument("--verbose", type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Initializing backtranslation pipeline...")
    pipeline = BackTranslationBaseline(
        device=args.device,
        similarity_threshold=args.similarity_threshold,
        save_dropped_path=args.save_dropped_path
    )
    print("Processing dataset...")
    pipeline.process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
