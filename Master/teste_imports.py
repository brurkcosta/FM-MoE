import sys, importlib
import torch.nn as nn

for name in sys.argv[1:]:
    importlib.import_module(name)
    print("importado:", name, flush=True)

for i in range(20000):
    m = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(20)])
    del m
    if i % 1000 == 0:
        print(i, flush=True)
print("FIM OK")
