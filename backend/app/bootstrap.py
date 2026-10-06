import sys
from pathlib import Path
app_path = str(Path(__file__).resolve().parent)
if app_path not in sys.path:
    sys.path.insert(0, app_path)
