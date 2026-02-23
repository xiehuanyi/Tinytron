import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tinytron.distributed import parallel_state


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        logits: [B, T_local, Vocab_Size]
        targets: [B, T_local]
        """
        logits = logits.view(-1, logits.size(-1)) # [B * T_local, V]
        targets = targets.view(-1)                # [B * T_local]

        unreduced_loss = F.cross_entropy(
            logits, targets, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )

        valid_mask = (targets != self.ignore_index).float()
        local_loss_sum = (unreduced_loss * valid_mask).sum()
        local_valid_count = valid_mask.sum()

        sp_group = parallel_state.get_sp_group()
        global_valid_count = local_valid_count.clone().detach()
        dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM, group=sp_group)
        global_valid_count = global_valid_count.clamp(min=1.0)
        loss = local_loss_sum / global_valid_count

        with torch.no_grad():
            global_loss_sum = local_loss_sum.clone().detach()
            dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=sp_group)
            logging_loss = global_loss_sum / global_valid_count # for logging

        return loss, logging_loss


class ExpertLoadBalancingLoss(nn.Module):
    def __init__(self, num_experts: int, top_k: int, alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha

    def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
        flat_logits = gate_logits.view(-1, self.num_experts)
        local_tokens = flat_logits.size(0)

        routing_probs = F.softmax(flat_logits, dim=-1)
        local_P_i_sum = routing_probs.sum(dim=0)  # [num_experts]

        _, selected_experts = torch.topk(flat_logits, self.top_k, dim=-1) # [local_tokens, top_k]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        local_f_i_sum = expert_mask.sum(dim=[0, 1])

        stats = torch.cat([local_P_i_sum, local_f_i_sum])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=parallel_state.get_ep_group())
        global_P_i_sum, global_f_i_sum = stats.chunk(2)
        global_tokens = local_tokens * parallel_state.get_ep_world_size()

        P_i = global_P_i_sum / global_tokens
        f_i = global_f_i_sum / (global_tokens * self.top_k)

        loss = self.alpha * self.num_experts * torch.sum(f_i * P_i)
        
        return loss