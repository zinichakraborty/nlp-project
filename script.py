from openai import OpenAI
import os
from dotenv import load_dotenv
import csv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_PATH = "data/processed/train_processed.csv"
OUTPUT_PATH = "data/processed/train_reordered_pairs.csv"

def main():
    with open(INPUT_PATH, "r") as f_in, open(OUTPUT_PATH, "w", newline="") as f_out:
        datareader = csv.reader(f_in)
        # ✅ quote all fields so every row looks the same
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)

        writer.writerow(["original_story", "reordered_story"])

        for i, row in enumerate(datareader):
            if i == 0:
                continue

            original_story = row[1]     # plain string
            shuffled_story = row[2]

            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Reorder these sentences chronologically keeping the same "
                            f"pipe-separated format: {shuffled_story}"
                        ),
                    }
                ],
            )

            reordered_story = response.choices[0].message.content.strip()

            writer.writerow([original_story, reordered_story])
            f_out.flush()                      # 🔥 force CSV flush to disk
            os.fsync(f_out.fileno())  

            if i % 50 == 0 or i == 1:
                print(f"Processed {i} rows...", flush=True)

if __name__ == "__main__":
    main()
