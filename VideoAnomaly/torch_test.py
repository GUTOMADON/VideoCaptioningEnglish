import importlib, sys
importlib.invalidate_caches()
print('python ok')
try:
    import torch
    print('torch', torch.__version__)
except Exception as e:
    print('torch import failed:', repr(e))
