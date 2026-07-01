"""Test setup: fake key so resolver's module-level client can init on import.

No real API/model calls are made — all externals are mocked in tests.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
