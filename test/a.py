import torch

a = torch.ones((4,10))
b = torch.norm(a.view(4, -1, 2), dim=-1)

c = torch.ones((4,))
d = c>0

breakpoint()