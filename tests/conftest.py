import pytest
import logging
from tests.util import train_skl, hls_convert, vhdl_convert, predict

@pytest.fixture(autouse=True)
def no_logs_gte_error(caplog):
    yield
    errors = [record for record in caplog.get_records('call') if record.levelno >= logging.ERROR]
    assert not errors