
from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter


class minADE(Metric):

    """最小平均位移误差（minADE）指标。

    在每个样本的多模态预测轨迹中，选取 top-k 候选，
    再按给定准则（FDE 或 ADE）挑选最优轨迹并累计 ADE。
    """

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minADE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        # 过滤无效样本/时间步，统一张量形状并返回有效掩码。
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        # 从多模态轨迹中取概率最高的 top-k 候选（或按默认策略取前 k 条）。
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == 'FDE':
            # 每个样本最后一个有效时间步索引，用于按 FDE 选择“最佳”候选轨迹。
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            # 在最后有效时刻，找到与 GT 终点距离最小的候选轨迹索引。
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
            # 使用选中的候选轨迹计算整段 ADE（仅对有效时间步平均），再对 batch 求和累计。
            self.sum += ((torch.norm(pred_topk[torch.arange(pred.size(0)), inds_best] - target, p=2, dim=-1) *
                          valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        elif min_criterion == 'ADE':
            # 直接按 ADE 选最优候选：先算每条候选在时间维上的误差和，再取最小值。
            self.sum += ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                          valid_mask.unsqueeze(1)).sum(dim=-1).min(dim=-1)[0] / valid_mask.sum(dim=-1)).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        # 记录样本数，用于最终取平均。
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        # 返回数据集级别的平均 minADE。
        return self.sum / self.count
