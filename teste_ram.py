import torch.nn as nn

for i in range(20000):
    m = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(20)])
    del m
    if i % 500 == 0:
        print(i, flush=True)
