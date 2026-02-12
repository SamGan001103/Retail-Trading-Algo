import sys
import os

# Set up paths like the scripts do
root_dir = os.path.abspath('.')
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'core'))
sys.path.insert(0, os.path.join(root_dir, 'bot'))
sys.path.insert(0, os.path.join(root_dir, 'account'))
sys.path.insert(0, os.path.join(root_dir, 'orders'))
sys.path.insert(0, os.path.join(root_dir, 'positions'))
sys.path.insert(0, os.path.join(root_dir, 'market'))
sys.path.insert(0, os.path.join(root_dir, 'tests'))

errors = []
imports_to_test = [
    'projectx_api',
    'execution_engine',
    'realtime_client',
    'market_lookup',
    'bot_runtime',
]

for module in imports_to_test:
    try:
        __import__(module)
        print(f'✓ {module}')
    except Exception as e:
        errors.append(f'{module}: {e}')
        print(f'✗ {module}: {e}')

if not errors:
    print('\n✅ All imports work correctly!')
else:
    print(f'\n❌ {len(errors)} import error(s)')
