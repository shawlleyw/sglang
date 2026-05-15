import time
import pytest
from sglang.srt.paras.migration_metrics import metrics, time_block, MigrationMetrics


@pytest.fixture(autouse=True)
def reset_metrics():
    # Reset module singleton before each test
    from sglang.srt.paras import migration_metrics as mm
    mm.metrics = MigrationMetrics()
    yield


def test_metrics_module_imports():
    from sglang.srt.paras.migration_metrics import metrics, time_block
    assert metrics is not None


def test_initial_state_zero():
    from sglang.srt.paras.migration_metrics import metrics
    assert metrics.failures_total == 0
    assert metrics.fallbacks_total == 0
    assert metrics.serialize_ms_ema == 0.0


def test_time_block_updates_ema():
    from sglang.srt.paras.migration_metrics import metrics, time_block
    with time_block("serialize_ms_ema"):
        time.sleep(0.01)  # ~10ms
    assert metrics.serialize_ms_ema > 5.0  # at least 5ms
    assert metrics.serialize_ms_ema < 100.0  # under 100ms sanity bound


def test_ema_converges():
    from sglang.srt.paras.migration_metrics import metrics
    metrics.update_ema("remap_ms_ema", 10.0)
    first = metrics.remap_ms_ema
    assert first == 10.0  # first sample
    metrics.update_ema("remap_ms_ema", 20.0)
    # alpha=0.2: new = 0.2*20 + 0.8*10 = 4 + 8 = 12
    assert 11.5 < metrics.remap_ms_ema < 12.5


def test_counter_increments():
    from sglang.srt.paras.migration_metrics import metrics
    metrics.failures_total += 1
    metrics.failures_total += 1
    assert metrics.failures_total == 2


def test_as_dict_export():
    from sglang.srt.paras.migration_metrics import metrics
    d = metrics.as_dict()
    assert "paras_radix_migration_failures_total" in d
    assert "paras_radix_migration_serialize_ms_ema" in d
    assert len(d) >= 5
