# -*- coding: utf-8 -*-
"""
loop3.py OOD泛化评测（原生难度）
模式1：目标导向（从训练集提取目标状态嵌入，反向回溯辅助决策）
模式2：随机基准（目标提取失败时自动降级）
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
MAX_STEPS = 50
GOAL_MODE = True  # 目标导向评测


def build_vocab(data):
    w2i = {w: i for i, w in enumerate(data.get("vocab", []))}
    a2i = {a: i for i, a in enumerate(data.get("actions", []))}
    return w2i, a2i


def tokenize(text, w2i, max_len=64):
    ids = [w2i.get(w.lower(), w2i.get("<unk>", 1)) for w in text.split()]
    ids = ids[:max_len]
    ids += [w2i.get("<pad>", 0)] * (max_len - len(ids))
    return ids


def load_model(data):
    meta = json.load(open(META_PATH, encoding="utf-8"))
    model = LoopWM(meta["vocab_size"], meta["action_size"], hidden_dim=meta["hidden_dim"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f"加载模型...")
    print(f"  词表：{meta['vocab_size']} | 动作：{meta['action_size']}")
    print(f"  模型参数量：{meta['n_params']:,}")
    return model, meta


def extract_goal_emb(model, data, w2i):
    """从训练集提取目标状态嵌入：取第一条成功轨迹的最终观测"""
    transitions = data["transitions"]
    # 取奖励>0 或 done 的样本作为目标状态
    for t in transitions:
        if t.get("reward", 0) > 0 or t.get("done", False):
            ids = torch.tensor([tokenize(t["next_obs_text"], w2i)], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                emb = model.encode(ids)
            print("  目标状态提取成功")
            return emb
    print("  警告：目标提取失败，将使用随机基准")
    return None


def load_env():
    from scienceworld import ScienceWorld
    env = ScienceWorld(envStepLimit=MAX_STEPS)
    return env


def get_actions(env):
    try:
        return list(env.valid_actions())
    except AttributeError:
        return list(env.get_valid_actions())


def step_env(env, action):
    try:
        obs, reward, done, info = env.step(action)
    except TypeError:
        obs, reward, done, info = env.step(action, 0)
    return obs, reward, done, info


def score_actions(model, obs_text, actions, w2i, a2i, goal_emb):
    """对候选动作打分，返回最优动作"""
    ids = torch.tensor([tokenize(obs_text, w2i)], dtype=torch.long, device=DEVICE)
    act_ids = torch.tensor([a2i.get(a, 0) for a in actions], dtype=torch.long, device=DEVICE)
    n = len(actions)
    act_onehot = torch.zeros(n, len(a2i), device=DEVICE)
    act_onehot.scatter_(1, act_ids.unsqueeze(-1), 1.0)
    obs_batch = ids.repeat(n, 1)
    with torch.no_grad():
        _, act_logits, _, _ = model(obs_batch, act_onehot=act_onehot,
                                    state_emb=goal_emb.repeat(n, 1) if goal_emb is not None else None)
    scores = act_logits[:, act_ids].diag()
    return actions[int(scores.argmax().item())]


def main():
    data = json.load(open(DATA_PATH, encoding="utf-8"))
    w2i, a2i = build_vocab(data)
    model, meta = load_model(data)

    mode = "目标导向" if GOAL_MODE else "随机基准"
    goal_emb = extract_goal_emb(model, data, w2i) if GOAL_MODE else None
    print(f"===== OOD泛化评测（原生难度，{mode}） ({N_EVAL}轮) =====")

    env = load_env()
    task_idx = data.get("task", "melt-1-2")
    try:
        env.load_task(task_idx, 0)
    except Exception:
        pass

    success = 0
    total_steps = 0
    for ep in range(1, N_EVAL + 1):
        try:
            obs = env.reset(task_idx, 0)
        except TypeError:
            obs = env.reset(task_idx, 0, 0)
        except Exception:
            obs = env.reset(task_idx, random.randint(0, 30))

        steps = 0
        done = False
        reward = 0.0
        for _ in range(MAX_STEPS):
            actions = get_actions(env)
            if not actions:
                break
            action = score_actions(model, obs, actions, w2i, a2i, goal_emb)
            obs, reward, done, _ = step_env(env, action)
            steps += 1
            if done or reward > 0:
                break

        total_steps += steps
        if done or reward > 0:
            success += 1
            print(f"  Ep {ep:2d}: ✅ 成功 | 步数：{steps:2d} | 奖励：{reward}")
        else:
            print(f"  Ep {ep:2d}: ❌ 失败 | 步数：{steps:2d} | 奖励：{reward}")

    print("=============================================")
    print("  OOD泛化评测 最终结果")
    print("=============================================")
    print(f"  测试轮数：{N_EVAL}")
    print(f"  成功轮数：{success}")
    print(f"  EM得分：{success / N_EVAL * 100:.2f}%")
    print(f"  总平均步数：{total_steps / N_EVAL:.1f}")
    if success:
        print(f"  成功局平均步数：{total_steps / success:.1f}")


if __name__ == "__main__":
    main()
