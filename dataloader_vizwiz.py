from datasets import load_dataset
ds = load_dataset('lmms-lab/VizWiz-VQA')

def majority_answer(answers):
    if not answers: return ''
    clean = []
    for a in answers:
        if isinstance(a, str): clean.append(a.strip().lower())
        elif isinstance(a, dict): clean.append(str(a.get('answer','')).strip().lower())
        else: clean.append(str(a).strip().lower())
    clean = [a for a in clean if a and a not in ('unanswerable','unsuitable')]
    return Counter(clean).most_common(1)[0][0] if clean else ''

all_samples = []
for s in tqdm(ds['val'], desc="VizWiz val"):
    try:
        img = s.get('image'); q = str(s.get('question',''))
        if not img or not q: continue
        answer = s.get('answer')
        if answer is not None: answer = str(answer).strip().lower()
        else: answer = majority_answer(s.get('answers', []))
        if not answer or answer in ('unanswerable','unsuitable',''): continue
        all_samples.append({'image': np.array(img.convert('RGB').resize((224,224)), dtype=np.float32)/255.0, 'question': q, 'answer': answer})
    except: continue
del ds; random.shuffle(all_samples)
sp = int(len(all_samples)*0.8)
train_samples, test_samples = all_samples[:sp], all_samples[sp:]
print(f"  Total usable: {len(all_samples)}, Train: {len(train_samples)}, Test: {len(test_samples)}")

all_ans = [s['answer'] for s in all_samples]; counts = Counter(all_ans)
filtered = [(a,c) for a,c in counts.most_common() if c >= MIN_ANS_FREQ][:MAX_ANS_VOCAB]
answer_vocab = {'<unk>': 0}
for i, (a,_) in enumerate(sorted(filtered, key=lambda x: x[0])): answer_vocab[a] = i+1
num_classes = len(answer_vocab)
cov = sum(counts[a] for a in answer_vocab if a in counts) / len(all_ans) * 100
print(f"  Vocab: {num_classes} classes, coverage: {cov:.1f}%")

indices = list(range(len(train_samples))); random.shuffle(indices)
n_val = int(len(indices)*VAL_SPLIT)
trn = [train_samples[i] for i in indices[n_val:]]; val = [train_samples[i] for i in indices[:n_val]]
print(f"  Train: {len(trn)}, Val: {len(val)}, Test: {len(test_samples)}")
