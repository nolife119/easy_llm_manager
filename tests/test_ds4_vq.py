"""CPU-only contract tests for the experimental DS4 adapter."""
import json
from pathlib import Path
import tempfile
import unittest
import ds4_vq

class Ds4VqTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "release.json").write_text(json.dumps({"format": "GVQ1"}), encoding="utf-8")
        for i in range(48):
            (self.root / f"expert-{i:02d}.gguf").touch()
        for name in ("body-nonexpert.gguf", "embed-tokens-bf16.gguf",
                     "mtp-q8_0.gguf", "vision-bf16.gguf",
                     "config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "processor_config.json"):
            (self.root / name).touch()
        self.exe = self.root / "ds4-server"
        self.exe.touch()

    def test_complete_directory(self):
        report = ds4_vq.inspect_release(self.root)
        self.assertTrue(report["ready_for_engine_validation"])
        self.assertEqual(report["part_count"], 52)

    def test_missing_release(self):
        (self.root / "release.json").unlink()
        with self.assertRaises(ValueError):
            ds4_vq.inspect_release(self.root)

    def test_missing_part(self):
        (self.root / "vision-bf16.gguf").unlink()
        self.assertFalse(ds4_vq.inspect_release(self.root)["ready_for_engine_validation"])

    def test_ds4_command(self):
        cmd = ds4_vq.build_command(self.exe, self.root, context=32768, mtp_draft=3)
        self.assertIn("--cuda", cmd)
        self.assertEqual(cmd[cmd.index("--mtp-draft") + 1], "3")
        self.assertNotIn("--n-gpu-layers", cmd)

    def test_public_binding_blocked(self):
        with self.assertRaises(ValueError):
            ds4_vq.build_command(self.exe, self.root, host="0.0.0.0")

if __name__ == "__main__":
    unittest.main()
