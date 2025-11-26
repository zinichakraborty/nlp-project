import csv
import sys

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = max_int // 2

def kendall_tau(gold_idx, pred_idx):
    C = 0
    D = 0
    n = len(gold_idx)

    for i in range(n):
        for j in range(i+1, n):
            if(gold_idx[i] < gold_idx[j] and pred_idx[i] < pred_idx[j]) or (gold_idx[i] > gold_idx[j] and pred_idx[i] > pred_idx[j]):
                C += 1
            else:
                D += 1
        
    total_pairs = n * (n-1) // 2
    if total_pairs == 0:
        return 0
    return (C - D) / total_pairs

def compute_kendall_tau(processed_csv, reordered_csv):
    taus = []
    dropped_rows_count = 0
    gold_stories = []
    with open(processed_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences = row['gold'].split(" | ")
            normalized_gold = [' '.join(s.split()) for s in sentences]
            gold_stories.append(normalized_gold)
        
    pred_stories = []
    with open(reordered_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = row['reordered_story'].split(' | ')
            pred_stories.append((pred, row))
    
    for gold, (pred, row) in zip(gold_stories, pred_stories):
        gold_idx = list(range(len(gold)))
        gold_map = {s: i for i, s in enumerate(gold)}
        try:
            pred_idx = [gold_map[s] for s in pred]
            tau = kendall_tau(gold_idx, pred_idx)
            taus.append(tau)
        except KeyError as e:
            dropped_rows_count += 1
            continue
    
    return taus, dropped_rows_count