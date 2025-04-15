import torch

def _random_router_with_weights(weighted_router: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    num_tokens = router_logits.shape[0]
    device = router_logits.device
    # first select the top0
    top0_ids = torch.zeros(num_tokens, dtype=torch.long, device=device)
    torch.multinomial(weighted_router[0], num_tokens, replacement=True, out=top0_ids)
    
    # mask the top0 elements
    mask = torch.ones_like(router_logits)
    mask[torch.arange(num_tokens), top0_ids] = 0
    
    # transform the conditional probabilities
    filtered_weight0 = weighted_router[0] * mask
    filtered_weight0 = filtered_weight0 / torch.sum(filtered_weight0, dim=1, keepdim=True)
    top1_weights = weighted_router[1] * mask * (1 + filtered_weight0)
    top1_weights = top1_weights / torch.sum(top1_weights, dim=1, keepdim=True)
    
    # then select the top1
    top1_ids = torch.zeros(num_tokens, dtype=torch.long, device=device)
    torch.multinomial(top1_weights, 1, replacement=True, out=top1_ids)
    top1_ids = top1_ids.squeeze(1)
    
    # change the router logits
    router_logits[torch.arange(num_tokens), top0_ids] = 5.0
    router_logits[torch.arange(num_tokens), top1_ids] = 3.0
    
    # the original values were in [0, 1]
    router_logits = router_logits / torch.sum(router_logits, dim=1, keepdim=True)
    return router_logits

weighted_router = torch.tensor(
    [
        [9, 10, 3, 4, 5, 6, 1, 8],
        [3, 4, 2, 2, 1, 1, 1, 1]
    ],
    device="cuda",
)
weighted_router = weighted_router / torch.sum(weighted_router, dim=1, keepdim=True)

bs = 1024
n_e = 8

max_sum = [0] * 8
sec_sum = [0] * 8

T = 100

for _ in range(T):

    router_logits = torch.randn((bs, n_e)).cuda()

    new_router_logits: torch.Tensor = _random_router_with_weights(weighted_router, router_logits)

    value, ids = new_router_logits.topk(2, dim=-1)
    
    ids = ids.tolist()

    for i in range(bs):
        max_sum[ids[i][0]] += 1
        sec_sum[ids[i][1]] += 1

max_sum = torch.Tensor(max_sum)
sec_sum = torch.Tensor(sec_sum)
print(max_sum / max_sum.min().item())
print(sec_sum / sec_sum.min().item())