import sys
import os

# Add subdirectories to Python path for all modules
project_root = os.path.dirname(os.path.abspath(__file__))
for subdir in ['core', 'bot', 'account', 'orders', 'positions', 'market', 'tests']:
    path = os.path.join(project_root, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)
