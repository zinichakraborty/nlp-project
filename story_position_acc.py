import csv

def count_sentence_mismatches(csv_path):
    mismatch_counts = [0, 0, 0, 0, 0]

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            orig = [s.strip() for s in row["original_story"].split("|")]
            reord = [s.strip() for s in row["reordered_story"].split("|")]

            for i in range(5):
                if orig[i] != reord[i]:
                    mismatch_counts[i] += 1

    return mismatch_counts

import csv

def get_sentence_2_3_mismatches(csv_path, k=10):
    examples = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if len(examples) >= k:
                break

            orig = [s.strip() for s in row["original_story"].split("|")]
            reord = [s.strip() for s in row["reordered_story"].split("|")]

            if len(orig) != 5 or len(reord) != 5:
                continue

            mismatch_2 = orig[1] != reord[1]
            mismatch_3 = orig[2] != reord[2]

            if mismatch_2 or mismatch_3:
                examples.append({
                    "orig":  [orig[1], orig[2]],
                    "reord": [reord[1], reord[2]]
                })

    return examples
