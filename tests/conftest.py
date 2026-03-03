"""Pytest configuration and fixtures."""
import os
import sys

# Ensure backend is importable when running tests from Voice-bot root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
