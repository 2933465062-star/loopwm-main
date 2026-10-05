# -*- coding: utf-8 -*-
"""
kernel/recurrent.py
轻量化双向循环世界模型（BiLoop）动力学核
- 前向循环通路：沿时序方向推演环境状态演化
- 反向回溯通路：从目标状态表示出发回溯推导当前状态
- 软门控融合：依据推理阶段动态调节两路信息信任度
"""
import torch
import torch.nn as nn
import math


class RecurrentDynamicsCore(nn.Module):
    """双向循环动力学核（BiLoop Core）"""

    def __init__(self, hidden_dim: int, action_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 动作嵌入（与词嵌入同维度，供融合使用）
        self.act_proj = nn.Linear(action_dim, hidden_dim, bias=False)

        # 前向循环单元（GRU 风格）
        self.fwd_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 反向回溯单元（同一组参数复用，实现参数共享的双向循环）
        self.bwd_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 软门控融合网络
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_act = nn.Sigmoid()

        # 残差 + 归一化
        self.ln = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs_emb: torch.Tensor, act_emb: torch.Tensor,
                state_emb: torch.Tensor = None, return_gate: bool = True):
        """
        obs_emb:   [B, H] 当前观测嵌入
        act_emb:   [B, A] 候选动作嵌入（one-hot 或 logits）
        state_emb: [B, H] 目标状态表示（反向回溯的锚点，可选）
        返回: h_out [B, H], 诊断字典
        """
        B, H = obs_emb.shape
        a = self.act_proj(act_emb)  # [B, H]

        # ---- 前向循环通路 ----
        h_f = self.fwd_gru(obs_emb + a)

        # ---- 反向回溯通路 ----
        if state_emb is None:
            h_b = torch.zeros_like(h_f)
        else:
            h_b = self.bwd_gru(state_emb + a)

        # ---- 软门控融合 ----
        gate = self.gate_act(self.gate_fc(torch.cat([h_f, h_b], dim=-1)))
        h_out = gate * h_f + (1 - gate) * h_b
        h_out = self.ln(h_out)

        diag = {}
        if return_gate:
            # 门控分布诊断（训练时用于检测门控退化）
            gate = gate.detach()
            diag['gate_std'] = gate.std().item()
            diag['gate_mean'] = gate.mean().item()
            diag['gate_sat'] = ((gate > 0.95) | (gate < 0.05)).float().mean().item()
        return h_out, diag


class LoopWM(nn.Module):
    """
    轻量化循环世界模型 LoopWM（BiLoop 结构）
    词嵌入 -> 轻量Transformer编码 -> 双向循环动力学核 -> 观测头/动作头
    """

    def __init__(self, vocab_size: int, action_size: int, hidden_dim: int = 128,
                 nhead: int = 4, enc_layers: int = 2, iter_steps: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.iter_steps = iter_steps

        # 词嵌入
        self.emb = nn.Embedding(vocab_size, hidden_dim)

        # 轻量 Transformer 编码器
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 动力学核（双向循环 + 软门控）
        self.core = RecurrentDynamicsCore(hidden_dim, action_size)

        # 观测头（预测下一时刻观测嵌入）
        self.obs_head = nn.Linear(hidden_dim, hidden_dim)

        # 动作判别头（对候选动作打分）
        self.act_head = nn.Linear(hidden_dim, action_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.emb.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, obs_ids: torch.Tensor):
        """观测文本 -> 观测嵌入 [B, H]"""
        x = self.emb(obs_ids)                      # [B, L, H]
        x = self.encoder(x)                        # [B, L, H]
        x = x.transpose(1, 2)                      # [B, H, L]
        h = self.pool(x).squeeze(-1)               # [B, H]
        return h

    def forward(self, obs_ids: torch.Tensor, act_ids: torch.Tensor = None,
                act_onehot: torch.Tensor = None, state_emb: torch.Tensor = None,
                num_iters: int = None):
        """前向推演：编码 -> 动力学核（多步迭代） -> 观测头/动作头"""
        h0 = self.encode(obs_ids)
        if act_onehot is None:
            act_onehot = torch.zeros(h0.shape[0], self.action_size, device=h0.device)
            if act_ids is not None:
                act_onehot.scatter_(1, act_ids.unsqueeze(-1), 1.0)

        iters = num_iters if num_iters is not None else self.iter_steps
        h = h0
        gate_stats = []
        for _ in range(iters):
            h, diag = self.core(h, act_onehot, state_emb=state_emb)
            gate_stats.append(diag)

        pred_emb = self.obs_head(h)
        act_logits = self.act_head(h)
        return pred_emb, act_logits, h, gate_stats
