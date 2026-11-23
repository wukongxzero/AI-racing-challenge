"""Fast, non-simulation sanity checks for the high-speed benchmark."""
import unittest
from .benchmark_high_speed import run_case

class HighSpeedBenchmarkTests(unittest.TestCase):
    def test_small_benchmark_and_telemetry(self):
        metrics, rows = run_case(6, steps=12)
        self.assertEqual(metrics["steps"], 12)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all("beta_deg" in r and "e_c" in r for r in rows))
        self.assertEqual(metrics["off_track_count"], 0)

if __name__ == "__main__":
    unittest.main()
