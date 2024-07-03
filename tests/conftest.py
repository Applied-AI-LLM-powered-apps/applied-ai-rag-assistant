# tests/conftest.py
import sys
import os

import pytest

from trulens_eval import (
    Tru
)


# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def trulens_prepare():
    tru = Tru(database_redact_keys=True)
    tru.reset_database()
    return tru