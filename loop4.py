# -*- coding: utf-8 -*-
"""
loop4.py Melt 评测（文本指标）
在训练集上评测：观测重建文本 vs 真实观测文本
指标：EM / Token F1 / BLEU-4 / Entity F1
"""
import json
import re
import random
import torch
from kernel.recurrent import LoopWM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "sw_melt_ood_train.json"
MODEL_PATH = "loopwm_melt_ood_baseline.pt"
META_PATH = "loopwm_melt_ood_baseline_meta.json"
N_EVAL = 20


def build_vocab(data):
    w2i = {w: i for i, w in enumerate(data.get("vocab", []))}
    a2i = {a: i for i, a in enumerate(data.get("actions", []))}
    return w2i, a2i


def tokenize(text, w2i, max_len=64):
    ids = [w2i.get(w.lower(), w2i.get("<unk>", 1)) for w in text.split()]
    ids = ids[:max_len]
    ids += [w2i.get("<pad>", 0)] * (max_len - len(ids))
    return ids


def init_nltk():
    """确保 nltk punkt_tab 可用（离线报错时尝试下载）"""
    import nltk
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        pass


def word_tokenize(text):
    import nltk
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    except LookupError:
        # 降级：简单分词
        return re.findall(r"[a-z0-9']+", text.lower())


def compute_text_metrics(pred_text, ref_text):
    """Token F1 / BLEU-4 / Entity F1"""
    pred_tokens = word_tokenize(pred_text)
    ref_tokens = word_tokenize(ref_text)

    # Token F1
    pred_set, ref_set = set(pred_tokens), set(ref_tokens)
    inter = len(pred_set & ref_set)
    p = inter / len(pred_set) if pred_set else 0.0
    r = inter / len(ref_set) if ref_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # BLEU-4（近似：4-gram 精确率 + 简洁惩罚）
    def ngrams(toks, n):
        return set(zip(*[toks[i:] for i in range(n)])) if len(toks) >= n else set()
    bleu = 0.0
    for n in range(1, 5):
        pn = len(ngrams(pred_tokens, n) & ngrams(ref_tokens, n)) / max(1, len(ngrams(pred_tokens, n)))
        bleu += pn / 4.0
    bp = min(1.0, len(pred_tokens) / max(1, len(ref_tokens)))
    bleu *= bp

    # Entity F1（剔除停用词）
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
                 "on", "at", "and", "or", "it", "you", "i", "we", "they", "he",
                 "she", "this", "that", "then", "now", "with", "for", "as", "by"}
    pred_ent = set(t for t in pred_tokens if t not in stopwords)
    ref_ent = set(t for t in ref_tokens if t not in stopwords)
    i2 = len(pred_ent & ref_ent)
    p2 = i2 / len(pred_ent) if pred_ent else 0.0
    r2 = i2 / len(ref_ent) if ref_ent else 0.0
    ent = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0.0

    return f1 * 100.0, bleu * 100.0, ent * 100.0


def retrieve_text(pred_emb, data, model, w2i):
    """预测嵌入 -> 最近邻观测文本"""
    emb_pool = []
    texts = []
    with torch.no_grad():
        for t in data["transitions"]:
            ids = torch.tensor([tokenize(t["next_obs_text"], w2i)], dtype=torch.long, device=DEVICE)
            e = model.encode(ids)
            emb_pool.append(e)
            texts.append(t["next_obs_text"])
    pool = torch.cat(emb_pool, dim=0)
    sims = torch.cosine_similarity(pred_emb, pool, dim=-1)
    return texts[int(sims.argmax().item())]


def main():
    init_nltk()
    print("加载模型：Melt")
    data = json.load(open(DATA_PATH, encoding="utf-8"))
    w2i, a2i = build_vocab(data)
    meta = json.load(open(META_PATH, encoding="utf-8"))
    model = LoopWM(meta["vocab_size"], meta["action_size"], hidden_dim=meta["hidden_dim"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    print(f"===== Melt 评测（20轮） =====")
    transitions = data["transitions"]
    random.seed(0)
    sample_ids = random.sample(range(len(transitions)), min(N_EVAL, len(transitions)))

    em_hits = 0
    f1s, bleus, ents = [], [], []
    for idx in sample_ids:
        t = transitions[idx]
        obs_ids = torch.tensor([tokenize(t["obs_text"], w2i)], dtype=torch.long, device=DEVICE)
        act_ids = torch.tensor([a2i.get(t["action_str"], 0)], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pred_emb, act_logits, _, _ = model(obs_ids, act_ids=act_ids)
        pred_text = retrieve_text(pred_emb, data, model, w2i)
        ref_text = t["next_obs_text"]

        # EM：预测动作是否命中
        pred_act = int(act_logits.argmax(-1).item())
        em_hits += (pred_act == act_ids.item())

        f1, bleu, ent = compute_text_metrics(pred_text, ref_text)
        f1s.append(f1)
        bleus.append(bleu)
        ents.append(ent)

    em = em_hits / len(sample_ids) * 100.0
    print(f"  EM: {em:.1f}%")
    print(f"  Token F1: {sum(f1s) / len(f1s):.1f}%")
    print(f"  BLEU-4: {sum(bleus) / len(bleus):.1f}%")
    print(f"  Entity: {sum(ents) / len(ents):.1f}%")


if __name__ == "__main__":
    main()
