# -*- coding: utf-8 -*-
"""
loop2.py 训练轻量化循环世界模型（BiLoop）
输入：sw_melt_ood_train.json
输出：loopwm_melt_ood_baseline.pt + _meta.json
"""
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from kernel.recurrent import LoopWM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "sw_melt_ood_train.json"
OUT_MODEL = "loopwm_melt_ood_baseline.pt"
OUT_META = "loopwm_melt_ood_baseline_meta.json"

HIDDEN_DIM = 128
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
MAX_LEN = 64


def build_vocab(data):
    vocab = data.get("vocab", [])
    w2i = {w: i for i, w in enumerate(vocab)}
    a2i = {a: i for i, a in enumerate(data.get("actions", []))}
    return w2i, a2i


def tokenize(text, w2i, max_len=MAX_LEN):
    ids = [w2i.get(w.lower(), w2i.get("<unk>", 1)) for w in text.split()]
    ids = ids[:max_len]
    ids += [w2i.get("<pad>", 0)] * (max_len - len(ids))
    return ids


class SWDataset(Dataset):
    def __init__(self, transitions, w2i, a2i):
        self.items = []
        for t in transitions:
            obs_ids = tokenize(t["obs_text"], w2i)
            next_ids = tokenize(t["next_obs_text"], w2i)
            act = a2i.get(t["action_str"], 0)
            self.items.append((obs_ids, act, next_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        obs_ids, act, next_ids = self.items[i]
        return {
            "obs_ids": torch.tensor(obs_ids, dtype=torch.long),
            "act_ids": torch.tensor(act, dtype=torch.long),
            "next_ids": torch.tensor(next_ids, dtype=torch.long),
        }


def collate(batch):
    obs_ids = torch.stack([b["obs_ids"] for b in batch])
    act_ids = torch.tensor([b["act_ids"] for b in batch])
    next_ids = torch.stack([b["next_ids"] for b in batch])
    return {"obs_ids": obs_ids, "act_ids": act_ids, "next_ids": next_ids}


def main():
    print(f"Device: {DEVICE}")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    transitions = data["transitions"]
    w2i, a2i = build_vocab(data)
    vocab_size = len(w2i)
    action_size = len(a2i)
    print(f"加载数据集：{DATA_PATH}")
    print(f"  转换数：{len(transitions)}")
    print(f"  词表：{vocab_size} | 动作：{action_size}")
    print(f"  隐藏维度：{HIDDEN_DIM}")

    model = LoopWM(vocab_size, action_size, hidden_dim=HIDDEN_DIM,
                   nhead=4, enc_layers=2, iter_steps=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{n_params:,}")

    ds = SWDataset(transitions, w2i, a2i)
    n_train = int(len(ds) * 0.9)
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    print("Epoch     Loss   ValMSE CosSim   Time")
    print("---------------------------------------------")

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batch = 0
        for batch in train_loader:
            obs_ids = batch["obs_ids"].to(DEVICE)
            act_ids = batch["act_ids"].to(DEVICE)
            next_ids = batch["next_ids"].to(DEVICE)

            with torch.no_grad():
                model.eval()
                next_emb = model.encode(next_ids)
                model.train()
            pred_emb, act_logits, _, _ = model(obs_ids, act_ids=act_ids)

            loss_mse = mse(pred_emb, next_emb)
            loss_act = ce(act_logits, act_ids)
            loss = loss_mse + 0.5 * loss_act
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        # 验证
        model.eval()
        val_mse = 0.0
        cos_sim = 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                obs_ids = batch["obs_ids"].to(DEVICE)
                act_ids = batch["act_ids"].to(DEVICE)
                next_ids = batch["next_ids"].to(DEVICE)
                next_emb = model.encode(next_ids)
                pred_emb, _, _, _ = model(obs_ids, act_ids=act_ids)
                val_mse += mse(pred_emb, next_emb).item()
                cos = nn.functional.cosine_similarity(pred_emb, next_emb, dim=-1).mean().item()
                cos_sim += cos
                nv += 1
        val_mse /= nv
        cos_sim /= nv
        dt = time.time() - t0
        print(f"{epoch:5d} {total_loss / n_batch:.6f} {val_mse:.6f} {cos_sim:.4f} {dt:.1f}s")

    print(f"✅ 收敛于 epoch {EPOCHS - 1}")
    torch.save(model.state_dict(), OUT_MODEL)
    meta = {
        "vocab_size": vocab_size,
        "action_size": action_size,
        "hidden_dim": HIDDEN_DIM,
        "n_params": n_params,
        "n_transitions": len(transitions),
        "epochs": EPOCHS,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"模型保存：{OUT_MODEL}")
    print(f"元数据保存：{OUT_META}")


if __name__ == "__main__":
    main()
