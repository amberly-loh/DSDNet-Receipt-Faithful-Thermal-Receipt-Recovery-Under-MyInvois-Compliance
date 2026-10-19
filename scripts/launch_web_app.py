from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.web_app import run

threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5055")).start()
run(ROOT / "configs" / "default.json")
