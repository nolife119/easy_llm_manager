"""Offline tests; no GPU or installed model required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_lab as lab
import experiment_sweep as sweep


class WindowsExperimentLabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.exe = self.folder / "llama-server.exe"
        self.exe.write_bytes(b"stub")
        self.model = self.folder / "model.gguf"
        self.model.write_bytes(b"stub")

    def profile(self):
        return {
            "backend": "llama", "executable": str(self.exe),
            "model": str(self.model), "port": 8080,
            "alias": "test-model",
            "args": ["--model", "{model}", "--port", "{port}", "--alias", "{alias}"],
            "env": {}
        }

    def test_command_vector_no_shell(self):
        cmd = lab.make_command(self.profile())
        self.assertEqual(cmd[0], str(self.exe))
        self.assertEqual(cmd[2], str(self.model))
        self.assertIn("8080", cmd)
        self.assertIn("test-model", cmd)

    def test_backend_restrictions(self):
        p = self.profile()
        p["backend"] = "arbitrary"
        with self.assertRaises(ValueError):
            lab.validate(p)
        p = self.profile()
        p["host"] = "0.0.0.0"
        with self.assertRaises(ValueError):
            lab.validate(p)

    def test_missing_executable_and_model(self):
        p = self.profile()
        p["executable"] = str(self.folder / "missing.exe")
        with self.assertRaises(ValueError):
            lab.make_command(p)
        p = self.profile()
        p["model"] = str(self.folder / "missing.gguf")
        with self.assertRaises(ValueError):
            lab.make_command(p)

    def test_vq_requires_release_directory(self):
        p = self.profile()
        p["backend"] = "ds4"
        p["model"] = str(self.folder)
        with self.assertRaises(ValueError):
            lab.make_command(p)
        (self.folder / "release.json").write_text("{}")
        self.assertEqual(lab.make_command(p)[0], str(self.exe))

    def test_sweep_requires_profiles(self):
        with patch.object(lab, "config", return_value={"profiles": {"good": self.profile()}}):
            with self.assertRaises(ValueError):
                sweep.begin(["unknown"])
            with self.assertRaises(ValueError):
                sweep.begin([])


if __name__ == "__main__":
    unittest.main()
