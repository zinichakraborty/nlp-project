import re
import random
import os
import csv

def read_stories(file_path):
    stories = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sentences = re.split(r'\.\s+', line)
            sentences = [s.strip().rstrip('.') for s in sentences if s.strip()] 
            if len(sentences) == 5:
                stories.append(sentences)
    return stories

def randomize_story(sentences, rng):
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    shuffled = [sentences[i] for i in indices]
    return shuffled

def preprocessing(train_path, test_path, out_dir='data/processed', seed=42):
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    for type, path in [('train', train_path), ('test', test_path)]:
        stories = read_stories(path)
        out_file = os.path.join(out_dir, f'{type}_processed.csv')

        with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['story_id', 'gold', 'shuffled'])
            for i, sents in enumerate(stories):
                shuffled = randomize_story(sents, rng)
                writer.writerow([
                    i,
                    ' | '.join(sents),
                    ' | '.join(shuffled)
                ])


