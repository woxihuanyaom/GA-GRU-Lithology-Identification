"""Spyder entry point: an independent synthetic test, not a field experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import main

if __name__ == '__main__':
    main(['--demo'])
