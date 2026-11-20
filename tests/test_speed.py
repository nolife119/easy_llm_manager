"""GPU-free regression tests for acceleration policy."""
import unittest
import speed

HELP = """--spec-type none,ngram-mod,ngram-simple,draft-dspark
--backend-sampling
--split-mode none,layer,row,tensor
--spec-draft-model FNAME
--spec-draft-n-max N
"""

class SpeedTests(unittest.TestCase):
    def setUp(self):
        self.d = {"quant":"UD-IQ1_M","context":524288,"kv_type":"f16","kv_location":"gpu",
                  "mode":"gpu","parallel":1,"batch":2048,"ubatch":256,"load_mode":"none",
                  "model_path":"local-model.gguf","vision":False}

    def test_only_advertised_features(self):
        self.assertEqual(len(speed.variants(self.d,"--split-mode layer")),1)
        self.assertTrue(speed.capabilities(HELP)["ngram_mod"])
        with self.assertRaises(ValueError):
            speed.options({**self.d,"spec_mode":"ngram-mod"},"--spec-type none")
        self.assertEqual(speed.options({**self.d,"spec_mode":"ngram-mod","backend_sampling":True},HELP),
                         ["--backend-sampling","--spec-type","ngram-mod"])

    def test_variants_preserve_model_and_context(self):
        variants = speed.variants(self.d,HELP,True)
        self.assertGreater(len(variants),5)
        for _,d in variants:
            for k,v in self.d.items():
                self.assertEqual(d[k],v)
        self.assertFalse(any(d["cuda_p2p"] for _,d in speed.variants(self.d,HELP)))

    def test_environment_clears_inherited_flags(self):
        env={"GGML_CUDA_P2P":"0","ggml_cuda_disable_graphs":"1","PATH":"test"}
        self.assertEqual(speed.environment(env,self.d),{"PATH":"test"})
        self.assertEqual(speed.environment(env,{**self.d,"cuda_p2p":True,"cuda_graphs":"off"}),
                         {"PATH":"test","GGML_CUDA_P2P":"1","GGML_CUDA_DISABLE_GRAPHS":"1"})

    def test_draft_requires_path_and_opt_in(self):
        with self.assertRaises(ValueError):
            speed.options({**self.d,"spec_mode":"draft-dspark"},HELP)
        d={**self.d,"draft_path":"draft.gguf"}
        self.assertNotIn("draft-dspark",[x["spec_mode"] for _,x in speed.variants(d,HELP)])
        self.assertIn("draft-dspark",[x["spec_mode"] for _,x in speed.variants(d,HELP,False,True)])
        self.assertIn("--spec-draft-model",speed.options({**d,"spec_mode":"draft-dspark"},HELP))

    def test_tensor_sampling_conflict(self):
        with self.assertRaises(ValueError):
            speed.validated({**self.d,"split_mode":"tensor","backend_sampling":True})

    def test_output_match_and_both_workloads_required(self):
        def sample(rate,h="hash"):
            return {"predicted_per_second":rate,"requested_output":128,"predicted_n":128,"output_hash":h}
        base={"code":sample(21),"general":sample(21)}
        both={"code":sample(70),"general":sample(51)}
        self.assertTrue(speed.assess(both,base,50)["goal_pass"])
        self.assertFalse(speed.assess({"code":sample(70),"general":sample(30)},base,50)["goal_pass"])
        self.assertFalse(speed.assess({"code":sample(70,"different"),"general":sample(51)},base,50)["match"])
        with self.assertRaises(ValueError):
            speed.assess({"code":sample(70)},base,50)

if __name__ == "__main__":
    unittest.main()
