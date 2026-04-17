import time
from pathlib import Path


def get_base_dir():
    try:
        return Path(__file__).resolve().parents[1]  # src/ -> project root
    except NameError:
        return Path.cwd().parent


def log(step, start):
    print(f"[{step}] done in {time.time() - start:.2f}s")