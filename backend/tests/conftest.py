"""Test-collection environment.

``app/main.py`` ends with a module-level ``app = create_app()``, and
``create_app`` now resolves ``get_settings()`` when no Settings is passed.
That runs at *import* time, so importing ``app.main`` raises RuntimeError
unless ANTHROPIC_API_KEY is set — and ``tests/test_health.py`` and
``tests/test_chat_endpoint.py`` both import it at module scope.

pytest imports conftest before collecting test modules, so setting a dummy
key here is what keeps collection working. ``setdefault`` means a real key
in the environment still wins, and tests that need a specific value (or its
absence) go on setting or deleting it themselves via monkeypatch.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-collection")
