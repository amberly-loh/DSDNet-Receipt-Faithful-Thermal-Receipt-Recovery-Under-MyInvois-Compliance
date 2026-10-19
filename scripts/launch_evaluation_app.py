from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.evaluation_ui import launch

launch(ROOT / "configs" / "default.json")
