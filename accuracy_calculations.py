import csv

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
    return (C - D) / total_pairs

def normalize(s):
    return " ".join(s.strip().split())

def compute_kendall_tau(reordered_csv):
    taus = []
    out_of_range_count = 0
    wrong_num_sentences_count = 0
    
    with open(reordered_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold = [normalize(s) for s in row['gold'].split(' | ')]
            pred = [normalize(s) for s in row['model_reordered'].split(' | ')]

            if any("INDEX_OUT_OF_RANGE" in s for s in pred):
                taus.append(0)
                out_of_range_count += 1
                continue

            if len(pred) != len(gold):
                taus.append(0)
                wrong_num_sentences_count += 1
                continue

            gold_idx = list(range(len(gold)))
            pred_idx = [gold.index(s) for s in pred]

            taus.append(kendall_tau(gold_idx, pred_idx))
    
    return taus, out_of_range_count, wrong_num_sentences_count

def compute_pmr(csv_path):
    total = 0
    matches = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            gold = normalize(row["gold"])
            pred = normalize(row["model_reordered"])
            if pred == gold:
                matches += 1
    return matches / total, matches, total