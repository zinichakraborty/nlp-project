from openai import OpenAI
import os
from dotenv import load_dotenv
import csv
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

INPUT_PATH = "data/processed/train_processed.csv"
OUTPUT_PATH = "data/processed/train_reordered_pairs_qwen3_numeric.csv"

def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_pipe_line(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = [l for l in lines if l.count("|") >= 2]

    if not candidates:
        return text.strip()

    candidates.sort(key=lambda s: s.count("|"), reverse=True)
    return candidates[0]

SYSTEM_PROMPT = "You are a strict sentence ordering function. Given 5 sentences separated by the '|' character, determine the correct chronological order."

def main():
    with open(INPUT_PATH, "r") as f_in, open(OUTPUT_PATH, "w", newline="") as f_out:
        datareader = csv.reader(f_in)
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)

        writer.writerow(["original_story", "reordered_story"])

        for i, row in enumerate(datareader):
            if i == 0:
                continue

            original_story = row[1]
            shuffled_story = row[2]

            response = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Input: {shuffled_story}\n\n"
                            "Output the 5-digit order only."
                        ),
                    },
                ],
            )


            raw = response.choices[0].message.content
            clean = remove_think_blocks(raw)
            reordered_story = extract_pipe_line(clean)
            reordered_story = ''.join(ch for ch in reordered_story if ch.isdigit())

            writer.writerow([original_story, reordered_story])
            f_out.flush()
            os.fsync(f_out.fileno())  

            if i % 50 == 0 or i == 1:
                print(f"Processed {i} rows...", flush=True)

if __name__ == "__main__":
    main()
