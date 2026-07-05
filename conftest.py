import os
import sys

# make sure `from app import app` works in tests/ regardless of how pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
