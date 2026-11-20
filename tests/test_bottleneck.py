import io
import unittest
import bottleneck as b

class BottleneckTest(unittest.TestCase):
    def test_cases(self):
        c={x["id"]:x for x in b.cases(524288,32768,1024,True)}
        self.assertEqual(c["B0"]["n"],4096)
        self.assertEqual(c["C"]["n"],32768)
        self.assertEqual(c["T"]["parent"],"H")
        self.assertEqual(c["A"]["parent"],"T")
        self.assertEqual(c["S"]["temp"],1)
        self.assertEqual(c["R"]["stream"],True)
        self.assertEqual(c["L"]["out"],1024)
        self.assertTrue(all(x["n"]+x["out"]+2048<=524288 for x in c.values()))
    def test_requests(self):
        c={x["id"]:x for x in b.cases(524288)}
        p,r=b.request(c["B0"],"test",[1,2],"local")
        self.assertEqual(p,"/completion")
        self.assertEqual(r["prompt"],[1,2])
        p,r=b.request(c["A"],"test",[1,2],"local")
        self.assertEqual(p,"/v1/chat/completions")
        self.assertTrue(r["stream"])
        self.assertEqual(r["tools"],[b.TOOL])
        self.assertFalse(r["cache_prompt"])
    def test_sse(self):
        data=b.read_response(io.BytesIO(b'data: {"content":"hello"}\n\ndata: {"timings":{"predicted_n":128,"predicted_per_second":30}}\n\ndata: [DONE]\n'),True)
        data["wall_s"]=5
        self.assertEqual(data["chunks"],1)
        self.assertEqual(b.measurements(data,{},128)["decode"],30)
        self.assertTrue(b.measurements(data,{},128)["complete"])
    def test_missing_timing(self):
        with self.assertRaises(ValueError):
            b.measurements({"usage":{"completion_tokens":128},"wall_s":5},{},128)
    def test_compare(self):
        rows=[{"id":"B0","status":"ok","decode":30,"valid":2,"min":29,"max":31},
              {"id":"C","label":"context","parent":"B0","status":"ok","decode":21,"valid":2,"min":20,"max":22}]
        result=b.compare(rows)[0]
        self.assertEqual(result["delta_pct"],-30)
        self.assertTrue(result["nonoverlap"])
    def test_short_output(self):
        with self.assertRaises(ValueError):b.summarize([{"complete":False,"decode":31}])
