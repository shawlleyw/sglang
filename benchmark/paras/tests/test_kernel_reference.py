"""Reference selection must use all participating GPUs and reject failed data."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_kernel_ablation import reference_bandwidth


class KernelReferenceTest(unittest.TestCase):
    def fixture(self, path, outbound, inbound, status="Passed"):
        tests = [
            {
                "name": name,
                "status": status,
                "bandwidth_matrix": [list(map(str, values))],
            }
            for name, values in [
                ("one_to_all_write_sm", outbound),
                ("all_to_one_write_sm", inbound),
            ]
        ]
        path.write_text(
            "tool preamble\n" + json.dumps({"nvbandwidth": {"testcases": tests}})
        )

    def test_best_sustained_for_each_gpu_then_shared_bottleneck(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / name for name in ("small.log", "large.log")]
            self.fixture(paths[0], [200, 300], [250, 210])
            self.fixture(paths[1], [240, 310], [260, 220])
            result = reference_bandwidth(paths, 2)
            self.assertEqual(result["reference_GBps_per_gpu"], 220)
            self.assertEqual(
                result["best_per_gpu_GBps"]["one_to_all_write_sm"], [240, 310]
            )

    def test_failed_or_missing_gpu_reference_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.log"
            self.fixture(path, [200, 300], [250, 210], status="Waived")
            with self.assertRaisesRegex(ValueError, "failed"):
                reference_bandwidth([path], 2)
            self.fixture(path, [200], [250])
            with self.assertRaisesRegex(ValueError, "2-GPU"):
                reference_bandwidth([path], 2)
        with self.assertRaisesRegex(ValueError, "No nvbandwidth"):
            reference_bandwidth([], 2)


if __name__ == "__main__":
    unittest.main()
