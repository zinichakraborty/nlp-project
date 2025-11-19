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

def compute_kendall_tau(processed_csv, reordered_csv):
    taus = []
    gold_stories = []
    with open(processed_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold = row['gold'].split(' | ')
            gold_stories.append(gold)
        
    pred_stories = []
    with open(reordered_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = row['reordered_story'].split(' | ')
            pred_stories.append(pred)
    
    for gold, pred in zip(gold_stories, pred_stories):
        gold_idx = list(range(len(gold)))
        pred_idx = [gold.index(s) for s in pred]
        tau = kendall_tau(gold_idx, pred_idx)
        taus.append(tau)
    
    return taus