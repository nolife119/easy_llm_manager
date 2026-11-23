"""CPU-only deterministic regression tests. No CUDA or model downloads needed."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mini_runtime as r
import mini_platform as p


class MiniRuntimeTests(unittest.TestCase):
    def test_model_exact_names(self):
        self.assertEqual(r.REPO, "unsloth/Inkling-2-Mini-GGUF")
        self.assertEqual(r.MODEL, "Inkling-2-Mini-UD-Q8_K_XL.gguf")
        self.assertEqual(r.DRAFT, "dspark-Inkling-2-Mini-Q8_0.gguf")
        self.assertEqual(r.PROJECTOR, "mmproj-F16.gguf")

    def test_validate_defaults(self):
        cfg = r.validate({})
        self.assertEqual(cfg["context"], 32768)
        self.assertEqual(cfg["parallel"], 1)
        self.assertTrue(cfg["dspark"])
        self.assertTrue(cfg["vision"])

    def test_context_1m_is_opt_in(self):
        cfg = r.validate({"context": 1048576})
        self.assertEqual(cfg["context"], 1048576)
        with self.assertRaises(ValueError):
            r.validate({"context": 2**30})

    def test_bad_parallel_and_unknown_keys(self):
        for patch_cfg in ({"parallel": 0}, {"parallel": 1024}, {"port": 8766},
                          {"gpu_index": -1}, {"context": "32768"}, {"extra": 1},
                          {"dspark": "false"}, {"kv": "q4_0"}):
            with self.subTest(patch_cfg=patch_cfg), self.assertRaises(ValueError):
                r.validate(patch_cfg)

    def test_build_command_no_silent_dspark_disable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "llama-server.exe").write_bytes(b"fake")
            for file in r.FILES:
                with (root / file).open("wb") as f:
                    f.write(b"GGUF" + b"\x00" * 100001)
            cfg = r.validate({"engine": str(root / "llama-server.exe"), "model_dir": str(root)})
            caps = {"dspark": True, "flash": True, "metrics": True,
                    "kv": True}
            cmd = r.build_command(cfg, caps)
            self.assertIn("--model-draft", cmd)
            self.assertIn("draft-dspark", cmd)
            self.assertIn("--mmproj", cmd)
            self.assertIn("--jinja", cmd)
            self.assertIn("--metrics", cmd)
            self.assertIn("127.0.0.1", cmd)
            self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "3")
            self.assertEqual(cmd[cmd.index("--parallel") + 1], "1")
            self.assertEqual(cmd[cmd.index("--ctx-size") + 1], "32768")

    def test_no_gpu_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "llama-server.exe").write_bytes(b"fake")
            for file in (r.MODEL, r.PROJECTOR):
                (root / file).write_bytes(b"GGUF" + b"\x00" * 100001)
            cfg = r.validate({"engine": str(root / "llama-server.exe"),
                              "model_dir": str(root), "dspark": False})
            cmd = r.build_command(cfg, {"flash": True, "metrics": True, "kv": True})
            self.assertNotIn("--model-draft", cmd)

    def test_missing_required_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "llama-server.exe").write_bytes(b"fake")
            cfg = r.validate({"engine": str(Path(d)/"llama-server.exe"), "model_dir": d})
            with self.assertRaisesRegex(ValueError, "모델 파일 없음"):
                r.build_command(cfg, {"flash": True})

    def test_metrics(self):
        sample = ("# HELP test\nllamacpp:prompt_tokens_total 400\n"
                  "llamacpp:prompt_seconds_total 2\n"
                  "llamacpp:tokens_predicted_total 500\n"
                  "llamacpp:tokens_predicted_seconds_total 10\n")
        x = r.parse_metrics(sample)
        self.assertEqual(r.rate(x, "llamacpp:prompt_tokens_total", "llamacpp:prompt_seconds_total"), 200)
        self.assertEqual(r.rate(x, "llamacpp:tokens_predicted_total", "llamacpp:tokens_predicted_seconds_total"), 50)

    def test_png_probe_payload_valid(self):
        img = p.png_sample()
        self.assertEqual(img[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(img), 30)


    def test_huggingface_model_info_uses_supported_signature(self):
        """Regression: HfApi.model_info does not accept repo_type."""
        import hashlib
        import sys
        import types
        from unittest.mock import Mock
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            blobs = {
                name: b"GGUF" + name.encode("utf-8") * 20000
                for name in r.FILES
            }
            metadata = {
                name: SimpleNamespace(
                    etag=hashlib.sha256(data).hexdigest(), size=len(data)
                )
                for name, data in blobs.items()
            }
            api = Mock()
            api.model_info.return_value = SimpleNamespace(sha="test-revision")
            hub = types.ModuleType("huggingface_hub")
            hub.HfApi = lambda: api
            hub.hf_hub_url = lambda repo_id, filename, revision: filename

            def fake_download(repo_id, filename, revision, local_dir):
                target = Path(local_dir) / filename
                target.write_bytes(blobs[filename])
                return str(target)

            hub.hf_hub_download = fake_download
            download_module = types.ModuleType("huggingface_hub.file_download")
            download_module.get_hf_file_metadata = lambda url: metadata[url]
            modules = {"huggingface_hub": hub, "huggingface_hub.file_download": download_module}
            with patch.dict(sys.modules, modules), patch.object(r, "ensure_hub", return_value=hub):
                cfg = r.validate({"model_dir": folder})
                out = r.download_models(cfg, lambda message: None,
                                        lambda *args: None)

            api.model_info.assert_called_once_with(r.REPO)
            self.assertEqual(out["revision"], "test-revision")
            self.assertTrue(r.is_verified(cfg))

    def test_config_save_and_reload(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"INKLING_MINI_HOME": d}):
            cfg = r.validate({"context": 65536, "spec_tokens": 4})
            r.save_config(cfg)
            loaded = r.load_config()
            self.assertEqual(loaded["context"], 65536)
            self.assertEqual(loaded["spec_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
