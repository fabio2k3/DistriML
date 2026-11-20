import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # permite `import tests.helpers...`
SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
