# to check if Python and some libraries are installed and working normally
import importlib, sys
# control import caches during development
importlib.invalidate_caches()
print('python ok')
try:
    # here try to import PyTorch and print its version when available
    import torch
    print('torch', torch.__version__)
except Exception as e:
    print('torch import failed:', repr(e))
