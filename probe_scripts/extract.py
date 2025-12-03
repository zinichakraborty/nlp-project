import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract sentence embeddings + position labels for probing."
    )
    parser.add_argument("--csv_path", type=str, default="data/final_outputs/train_qwen_reordered.csv")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--text_column", type=str, default="gold")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_stories", type=int, default=400)
    return parser.parse_args()


def load_sentences_and_labels(csv_path: Path, text_column: str, max_stories: int | None):
    sentences = []
    labels = []
    story_ids = []
    sentence_indices = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_stories is not None and i >= max_stories:
                break

            story_id = int(row["story_id"])
            story_text = row[text_column]
            parts = [s.strip() for s in story_text.split("|")]
            if len(parts) != 5:
                raise ValueError(
                    f"Expected 5 sentences per story, got {len(parts)} for story_id={story_id}"
                )
            for idx, sent in enumerate(parts):
                sentences.append(sent)
                labels.append(idx)
                story_ids.append(story_id)
                sentence_indices.append(idx)

    return sentences, np.array(labels, dtype=np.int64), np.array(story_ids, dtype=np.int64), np.array(sentence_indices, dtype=np.int64)

def mean_pool_last_layer(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    masked_hidden = hidden_states * mask
    summed = masked_hidden.sum(dim=1)
    lengths = mask.sum(dim=1)
    lengths = torch.clamp(lengths, min=1)
    return summed / lengths


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)

    print(f"Loading sentences from {csv_path} (column: {args.text_column})...")
    sentences, labels, story_ids, sentence_indices = load_sentences_and_labels(
        csv_path, args.text_column, args.max_stories
    )
    print(f"Loaded {len(sentences)} sentences.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {args.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    all_embs = []
    batch_size = args.batch_size

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            end = start + batch_size
            batch_sents = sentences[start:end]
            inputs = tokenizer(
                batch_sents,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            print(f"Starting batch {start} to {end}", flush=True)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]

            sent_embs = mean_pool_last_layer(last_hidden, inputs["attention_mask"])
            sent_embs = sent_embs.to(torch.float32)

            all_embs.append(sent_embs.cpu().numpy())


            if (start // batch_size) % 10 == 0:
                print(f"Processed {end}/{len(sentences)} sentences...", flush=True)

    X = np.vstack(all_embs)
    y = labels

    print(f"Embeddings shape: {X.shape}, labels shape: {y.shape}")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        X=X,
        y=y,
        story_ids=story_ids,
        sentence_indices=sentence_indices,
    )
    print(f"Saved embeddings + labels to {output_path}")


if __name__ == "__main__":
    main()
