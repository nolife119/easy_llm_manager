"""Offline regression tests: python -m unittest discover -s tests -v.
No GPU, model downloads, server process or network required.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

_state = tempfile.TemporaryDirectory()
with patch.dict(os.environ, {"LOCALAPPDATA": _state.name}):
    import manager as m


class SmartFitTest(unittest.TestCase):
    def make(self, **changes):
        d = {**m.BASE["defaults"], "profile": "inkling2", "quant": "UD-IQ1_M",
             "purpose": "coding", "context": 131072, "parallel": 1, "unified": False,
             "mode": "gpu", "kv_location": "gpu", "vision": False,
             "gpu_ids": "", "split": "", "recovery_options": "", **changes}
        d["pool"] = d["context"] * d["parallel"]
        return d

    def hardware(self, free=90, ram=220):
        return ([{"index": i, "free": free, "total": 94} for i in (0, 1)], 256, ram)

    def search(self, d, hardware, strategy="gpu", precision="auto"):
        p, q = m.profile(d)
        return m.smart_search(d, p, q["gb"] * 1e9 / m.GIB, hardware, strategy, precision,
                              "balanced", {"cpu": True, "dynamic": False, "quant_kv": True})

    def test_tight_all_gpu_keeps_context(self):
        d = self.make()
        with patch.object(m, "WIN", True):
            out, near, count = self.search(d, self.hardware())
        self.assertGreater(count, 1)
        self.assertTrue(out)
        for item in out:
            s = item["settings"]
            self.assertEqual(s["context"], 131072)
            self.assertEqual(s["cpu_layers"], 0)
            self.assertEqual(s["kv_location"], "gpu")
            self.assertGreaterEqual(item["gpu_gap"], -1e-6)

    def test_impossible_gpu_returns_shortfall_without_offload(self):
        d = self.make()
        out, near, count = self.search(d, self.hardware(70))
        self.assertFalse(out)
        self.assertIsNotNone(near)
        self.assertGreater(near["deficit"], 0)
        self.assertEqual(near["settings"]["cpu_layers"], 0)

    def test_windows_mapping_risk_is_not_private_ram(self):
        d = self.make()
        _, q = m.profile(d)
        weight = q["gb"] * 1e9 / m.GIB
        with patch.object(m, "WIN", True):
            a = m.estimate({**d, "load_mode": "auto"}, self.hardware(), weight, fast=True)
            b = m.estimate({**d, "load_mode": "none"}, self.hardware(), weight, fast=True)
        self.assertAlmostEqual(a["mapped_extra_risk"], weight)
        self.assertAlmostEqual(a["host_peak_risk"] - a["host_ram"], weight)
        self.assertEqual(b["mapped_extra_risk"], 0)
        self.assertAlmostEqual(a["gpu"], b["gpu"])

    def test_long_context_vision_and_slots_remain_locked(self):
        d = self.make(context=1048576, purpose="long", vision=True)
        out, near, _ = self.search(d, self.hardware(), "auto")
        self.assertTrue(out)
        for item in out:
            s = item["settings"]
            self.assertEqual(s["context"], 1048576)
            self.assertEqual(s["parallel"], 1)
            self.assertTrue(s["vision"])
            self.assertEqual(s["kv_location"], "gpu")

    def test_full_swa_option_changes_estimate(self):
        d = self.make(kv_type="q8_0", recovery_options='{"--swa-full":true}')
        p, q = m.profile(d)
        x = m.estimate(d, self.hardware(), q["gb"]*1e9/m.GIB, fast=True)
        self.assertAlmostEqual(x["kv"], x["kv_worst"])

    def test_failure_diagnosis(self):
        self.assertEqual(m.startup_failure(["CUDA error: out of memory", "failed to create context"])["kind"], "memory")
        self.assertEqual(m.startup_failure(["failed to create context"])["kind"], "context")

    def test_gpu_max_preserves_quant_and_all_gpu(self):
        d = self.make(context=1048576, purpose="gpu_max", vision=False)
        p, q = m.profile(d)
        caps = {"cpu": True, "dynamic": True, "quant_kv": True}
        out, near, n = m.gpu_context_search(d, p, q["gb"]*1e9/m.GIB, self.hardware(), "auto", "balanced", caps)
        self.assertTrue(out)
        self.assertGreater(n, 1)
        self.assertEqual(out[0]["settings"]["context"], max(c["settings"]["context"] for c in out))
        for candidate in out:
            k = candidate["settings"]
            self.assertEqual(k["quant"], "UD-IQ1_M")
            self.assertEqual(k["mode"], "gpu")
            self.assertEqual(k["kv_location"], "gpu")
            self.assertEqual(k["parallel"], 1)
            self.assertEqual(k["ram_cache"], 0)
            self.assertEqual(k["cpu_layers"], 0)
            self.assertTrue(k["gpu_fast"])

    def test_gpu_max_impossible_does_not_offload(self):
        d = self.make(purpose="gpu_max")
        p, q = m.profile(d)
        out, near, _ = m.gpu_context_search(d, p, q["gb"]*1e9/m.GIB, self.hardware(70), "auto", "balanced",
                                            {"cpu": True, "dynamic": True, "quant_kv": True})
        self.assertFalse(out)
        self.assertGreater(near["deficit"], 0)
        self.assertEqual(near["settings"]["cpu_layers"], 0)

    def test_gpu_fast_disallows_cpu_kv_override(self):
        d = self.make(gpu_fast=True, recovery_options='{"--no-kv-offload":true}')
        with self.assertRaises(ValueError):
            m.validate(d)

    def test_download_benchmark_isolation(self):
        with patch.dict(m.TASKS, {"download": {"active": True}}, clear=True):
            with self.assertRaises(ValueError):
                m.start_task("benchmark", lambda: None)
        with patch.dict(m.TASKS, {"benchmark": {"active": True}}, clear=True):
            with self.assertRaises(ValueError):
                m.start_task("download", lambda: None)


if __name__ == "__main__":
    unittest.main()
