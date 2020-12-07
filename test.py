from opt_einsum.backends import torch

aa = torch.zeros((1, 12))
aa[0][4] = 1
print(aa)

bb = torch.zeros((1,12))
bb[0][3]=1
print(bb)

print(torch.sum(aa==bb))
print(torch.sum(aa==bb) / aa.view(-1).shape[0])
print(torch.sum(aa==bb, dtype=aa.dtype) / aa.view(-1).shape[0])

