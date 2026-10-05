# -*- coding: utf-8 -*-
"""
loop1.py 训练集生成（ScienceWorld）
任务：融化 1-2（默认），可切换为 烧水1-1 / 冻结1-3
输出：sw_melt_ood_train.json
"""
import json
import random
import os

random.seed(42)

TASK_ID = os.environ.get("SW_TASK", "melt-1-2")
N_EPISODES = int(os.environ.get("N_EPISODES", "100"))
MAX_STEPS = 50
OUT_PATH = os.environ.get("OUT_PATH", "sw_melt_ood_train.json")

# 各任务的专家关键词（优先执行包含目标词的动作）
EXPERT_HINTS = {
    "boil-1-1": ["boil", "heat", "use", "turn on", "wait"],
    "melt-1-2": ["melt", "heat", "use", "turn on", "wait"],
    "freeze-1-3": ["freeze", "put", "use", "turn on", "wait"],
}
hints = EXPERT_HINTS.get(TASK_ID, ["use", "wait", "pick", "put"])


def load_env():
    """加载 ScienceWorld 环境（兼容 0.9+ 与旧版接口）"""
    try:
        from scienceworld import ScienceWorld
    except ImportError:
        raise RuntimeError(
            "未安装 scienceworld 包，请先执行: pip install scienceworld，"
            "或在已装包的环境中运行本脚本。"
        )
    env = ScienceWorld(envStepLimit=MAX_STEPS)
    return env


def reset_env(env, task_idx: int, variation: int):
    """reset 并返回初始观测（兼容不同版本 API）"""
    try:
        obs = env.reset(task_idx, variation)
    except TypeError:
        obs = env.reset(task_idx, variation, 0)
    return obs


def get_valid_actions(env):
    try:
        return list(env.valid_actions())
    except AttributeError:
        return list(env.get_valid_actions())


def step_env(env, action: str):
    try:
        obs, reward, done, info = env.step(action)
    except TypeError:
        obs, reward, done, info = env.step(action, 0)
    return obs, reward, done, info


def expert_action(actions):
    """简单专家策略：优先命中目标关键词的动作，否则随机"""
    for h in hints:
        for a in actions:
            if h in a.lower():
                return a
    return random.choice(actions)


def main():
    print(f"正在生成 OOD训练集（原生难度）... 任务: {TASK_ID}")
    env = load_env()

    # 任务编号映射
    task_idx = TASK_ID
    try:
        env.load_task(task_idx, 0)
    except Exception:
        pass

    # 环境动作总数
    acts_all = get_valid_actions(env)
    print(f"环境动作总数：{len(acts_all)}")

    transitions = []
    success_episodes = 0
    variation = 0

    for ep in range(N_EPISODES):
        try:
            obs = reset_env(env, task_idx, variation)
        except Exception:
            # 部分版本不支持指定 variation，随机重置
            obs = reset_env(env, task_idx, random.randint(0, 30))

        ep_success = False
        prev_obs = obs  # 关键修复：记录初始观测，保证评测状态在训练集内
        for step in range(MAX_STEPS):
            actions = get_valid_actions(env)
            if not actions:
                break
            action = expert_action(actions)
            next_obs, reward, done, info = step_env(env, action)

            transitions.append({
                "obs_text": prev_obs,
                "action_str": action,
                "next_obs_text": next_obs,
                "reward": float(reward),
                "done": bool(done),
            })
            prev_obs = next_obs

            if done or reward > 0:
                ep_success = True
                break

        if ep_success:
            success_episodes += 1
        if (ep + 1) % 20 == 0:
            print(f"  进度 {ep + 1}/{N_EPISODES} | 成功 {success_episodes} 轮")

    # 词表与动作集合
    vocab = set(["<pad>", "<unk>"])
    actions = set()
    for t in transitions:
        for w in t["obs_text"].split():
            vocab.add(w.lower())
        for w in t["next_obs_text"].split():
            vocab.add(w.lower())
        actions.add(t["action_str"])

    expert_rate = success_episodes / N_EPISODES
    data = {
        "task": TASK_ID,
        "variation": variation,
        "n_episodes": N_EPISODES,
        "expert_success_rate": expert_rate,
        "transitions": transitions,
        "vocab": sorted(vocab),
        "actions": sorted(actions),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    print(f"✅ 训练集生成完成：{OUT_PATH}")
    print(f"  专家成功率：{expert_rate * 100:.1f}%")
    print(f"  有效转换数：{len(transitions)} 条")
    print(f"  词表大小：{len(vocab)}")
    print(f"  动作数：{len(actions)}")


if __name__ == "__main__":
    main()
