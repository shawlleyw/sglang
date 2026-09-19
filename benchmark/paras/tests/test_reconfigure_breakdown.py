"""CPU-only checks for additive accounting of nested switch phase timers."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconfigure.breakdown import breakdown_trial


class BreakdownTest(unittest.TestCase):
    def trial(self):
        return {
            "method": "naive_nccl",
            "direction": "ep_to_tp",
            "status": "passed",
            "switch_ms": 100,
            "rank_results": [
                {
                    "rank": 0,
                    "switch_ms": 95,
                    "phases": {
                        "weight_transfer_ms": 30,
                        "runtime_switch_ms": 35,
                        "discard_graphs_ms": 10,
                        "graph_capture_ms": 50,
                    },
                },
                {
                    "rank": 1,
                    "switch_ms": 100,
                    "phases": {
                        "weight_transfer_ms": 20,
                        "runtime_switch_ms": 30,
                        "discard_graphs_ms": 5,
                        "graph_capture_ms": 45,
                    },
                },
            ],
        }

    def test_uses_total_critical_rank_without_maximizing_each_phase(self):
        row = self.trial()
        original = copy.deepcopy(row)
        result = breakdown_trial(row)
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["weights_ms"], 20)
        self.assertEqual(result["graph_ms"], 50)
        self.assertEqual(result["others_ms"], 30)
        self.assertEqual(result["runtime_nonweight_ms"], 10)
        self.assertEqual(result["outside_phases_ms"], 20)
        self.assertEqual(
            sum(result[k] for k in ("weights_ms", "graph_ms", "others_ms")), 100
        )
        self.assertEqual(row, original)

    def test_host_auxiliary_enqueue_is_counted_once(self):
        row = self.trial()
        row["method"] = "host_reload"
        for rank in row["rank_results"]:
            rank["phases"]["auxiliary_reload_ms"] = 3
        result = breakdown_trial(row)
        self.assertEqual(result["weights_ms"], 23)
        self.assertEqual(result["others_ms"], 27)
        self.assertEqual(result["outside_phases_ms"], 17)

    def test_full_retains_graphs_and_has_runtime_remainder(self):
        row = self.trial()
        row["method"] = "full"
        for rank in row["rank_results"]:
            del rank["phases"]["discard_graphs_ms"]
            del rank["phases"]["graph_capture_ms"]
        result = breakdown_trial(row)
        self.assertEqual(result["graph_ms"], 0)
        self.assertEqual(result["others_ms"], 80)

    def test_restart_split_is_unavailable_not_all_others(self):
        result = breakdown_trial(
            {
                "method": "rebuild",
                "direction": "tp_to_ep",
                "status": "passed",
                "switch_ms": 200,
                "phases": {"engine_shutdown_ms": 10, "launch_and_initialize_ms": 190},
            },
            reference=True,
        )
        self.assertFalse(result["breakdown_available"])
        self.assertIsNone(result["weights_ms"])
        self.assertIsNone(result["graph_ms"])
        self.assertIsNone(result["others_ms"])
        self.assertEqual(result["total_ms"], 200)

    def test_rejects_inconsistent_nested_or_total_timers(self):
        row = self.trial()
        row["rank_results"][1]["phases"]["runtime_switch_ms"] = 10
        with self.assertRaisesRegex(ValueError, "enclosing duration"):
            breakdown_trial(row)
        row = self.trial()
        row["switch_ms"] = 101
        with self.assertRaisesRegex(ValueError, "maximum rank"):
            breakdown_trial(row)

    def test_failed_or_incomplete_trials_cannot_be_reported_as_zero_cost(self):
        row = self.trial()
        row["status"] = "failed"
        with self.assertRaises(ValueError):
            breakdown_trial(row)
        row = self.trial()
        del row["rank_results"][1]["phases"]["graph_capture_ms"]
        with self.assertRaises(KeyError):
            breakdown_trial(row)


if __name__ == "__main__":
    unittest.main()
