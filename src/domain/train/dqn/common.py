"""DQN 训练共享计算。 / Shared DQN training calculations."""

from __future__ import annotations


def masked_next_values(torch, target_model, next_states, legal_next_actions_batch, device: str):
    """只在下一状态的合法动作中取最大 Q 值。 / Maximize next-state Q values over legal actions only."""
    with torch.no_grad():
        q_values = target_model(next_states)
        values = []
        for row, legal_indexes in zip(q_values, legal_next_actions_batch):
            if legal_indexes:
                indexes = torch.tensor(legal_indexes, dtype=torch.long, device=device)
                values.append(row.index_select(0, indexes).max())
            else:
                values.append(torch.tensor(0.0, device=device))
        return torch.stack(values)
