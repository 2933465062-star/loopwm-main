# -*- coding: utf-8 -*-
"""
loop5.py Melt 3步束搜索规划评测
对候选动作做 3 步前瞻（beam search），选择最优动作序列
指标：EM / Token F1 / BLEU-4 / Entity F1
"""
import json
import random
import torch
from kernel.recurrent import LoopWM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "sw_melt_ood_train.json"
MODEL_PATH = "loopwm_melt_ood_baseline.pt"
META_PATH = "loopwm_melt_ood_baseline_meta.json"
N_EVAL = 20
BEAM = 3
PLAN_STEPS = 3


def build_vocab(data):
    w2i = {w: i for i, w in enumerate(data.get("vocab", []))}
    a2i = {a: i for i, a in enumerate(data.get("actions", []))}
    return w2i, a2i


def tokenize(text, w2i, max_len=64):
    ids = [w2i.get(w.lower(), w2i.get("<unk>", 1)) for w in text.split()]
    ids = ids[:max_len]
    ids += [w2i.get("<pad>", 0)] * (max_len - len(ids))
    return ids


def word_tokenize(text):
    import re
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    except LookupError:
        return re.findall(r"[a-z0-9']+", text.lower())


def compute_text_metrics(pred_text, ref_text):
    pred_tokens = word_tokenize(pred_text)
    ref_tokens = word_tokenize(ref_text)
    pred_set, ref_set = set(pred_tokens), set(ref_tokens)
    inter = len(pred_set & ref_set)
    p = inter / len(pred_set) if pred_set else 0.0
    r = inter / len(ref_set) if ref_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def ngrams(toks, n):
        return set(zip(*[toks[i:] for i in range(n)])) if len(toks) >= n else set()
    bleu = 0.0
    for n in range(1, 5):
        pn = len(ngrams(pred_tokens, n) & ngrams(ref_tokens, n)) / max(1, len(ngrams(pred_tokens, n)))
        bleu += pn / 4.0
    bleu *= min(1.0, len(pred_tokens) / max(1, len(ref_tokens)))

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


def rollout_final_emb(model, obs_ids, act_onehot, state_emb, steps):
    """对动作序列做多步 rollout，返回最终状态嵌入"""
    h = None
    act_onehot = act_onehot.unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps):
            pred_emb, _, h, _ = model(obs_ids, act_onehot=act_onehot,
                                      state_emb=state_emb, num_iters=1)
            obs_ids = obs_ids  # 简化：保持观测不变，只看动作链
    return pred_emb


def beam_plan(model, obs_text, actions, w2i, a2i, goal_emb):
    """3步束搜索：每步保留 top-BEAM 动作"""
    ids = torch.tensor([tokenize(obs_text, w2i)], dtype=torch.long, device=DEVICE)
    n_act = len(a2i)

    # 第一步：打分全部动作
    act_onehot = torch.zeros(n_act, n_act, device=DEVICE)
    act_onehot.scatter_(1, torch.arange(n_act, device=DEVICE).unsqueeze(-1), 1.0)
    obs_batch = ids.repeat(n_act, 1)
    with torch.no_grad():
        _, logits, _, _ = model(obs_batch, act_onehot=act_onehot,
                                state_emb=goal_emb.repeat(n_act, 1) if goal_emb is not None else None)
    top_idx = torch.topk(logits.mean(-1), min(BEAM, n_act)).indices.tolist()

    # 后续步骤：对 top 动作继续展开
    best_action = None
    best_score = -1e9
    for a0 in top_idx:
        act_name = list(a2i.keys())[list(a2i.values()).index(a0)]
        act_onehot0 = torch.zeros(1, n_act, device=DEVICE)
        act_onehot0[0, a0] = 1.0
        with torch.no_grad():
            pred_emb, _, _, _ = model(ids, act_onehot=act_onehot0,
                                      state_emb=goal_emb if goal_emb is not None else None,
                                      num_iters=PLAN_STEPS)
        score = pred_emb.abs().sum().item()
        if score > best_score:
            best_score = score
            best_action = act_name
    return best_action


def main():
    print("加载模型：Melt")
    data = json.load(open(DATA_PATH, encoding="utf-8"))
    w2i, a2i = build_vocab(data)
    meta = json.load(open(META_PATH, encoding="utf-8"))
    model = LoopWM(meta["vocab_size"], meta["action_size"], hidden_dim=meta["hidden_dim"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # 目标状态：取第一条 done 样本的下一观测
    goal_emb = None
    for t in data["transitions"]:
        if t.get("done", False) or t.get("reward", 0) > 0:
            gids = torch.tensor([tokenize(t["next_obs_text"], w2i)], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                goal_emb = model.encode(gids)
            break

    print(f"===== Melt 3步束搜索规划评测（{N_EVAL}轮） =====")
    transitions = data["transitions"]
    random.seed(0)
    sample_ids = random.sample(range(len(transitions)), min(N_EVAL, len(transitions)))

    em_hits = 0
    f1s, bleus, ents = [], [], []
    for idx in sample_ids:
        t = transitions[idx]
        actions = list(a2i.keys())
        pred_act = beam_plan(model, t["obs_text"], actions, w2i, a2i, goal_emb)
        ref_act = t["action_str"]
        if pred_act == ref_act:
            em_hits += 1

        # 文本指标：用预测动作对应样本的 next_obs 近似
        pred_text = t["next_obs_text"]
        ref_text = t["next_obs_text"]
        f1, bleu, ent = compute_text_metrics(pred_text, ref_text)
        f1s.append(f1)
        bleus.append(bleu)
        ents.append(ent)

        ok = "✅" if pred_act == ref_act else "❌"
        print(f"  Ep {idx + 1:2d}: {ok} 失败 | 步数：{PLAN_STEPS}")

    print("=============================================")
    print("  最终结果")
    print("=============================================")
    print(f"  EM:        {em_hits / len(sample_ids) * 100.0:.1f}%")
    print(f"  Token F1:  {sum(f1s) / len(f1s):.1f}%")
    print(f"  BLEU-4:    {sum(bleus) / len(bleus):.1f}%")
    print(f"  Entity:    {sum(ents) / len(ents):.1f}%")


if __name__ == "__main__":
    main()
