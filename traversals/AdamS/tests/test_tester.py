# tests/test_tester.py
import os, json, shutil, tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tester import OptimizerComparisonTester, OptimizerConfig
from v1 import TraversalAdamW

def tiny_dataset(n=256, d=32, k=4):
    X = torch.randn(n, d)
    W = torch.randn(d, k)
    y = (X @ W).argmax(1)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=64, shuffle=True), DataLoader(ds, batch_size=128, shuffle=False)

class TinyNet(nn.Module):
    def __init__(self, d=32, k=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, k))
    def forward(self, x): return self.net(x)

def test_instantiation_and_run_produces_json_and_tb_events():
    train, val = tiny_dataset()
    model_fn = lambda: TinyNet()
    tester = OptimizerComparisonTester(model_fn, train, val, eval_every=2)

    candidate_cfg = OptimizerConfig(
        name="trav_none_all",
        cls_path="v1.TraversalAdamW",
        kwargs=dict(traversal_mode="none", moment1_source="trav", moment2_source="trav", lr=1e-3),
    )

    tmpdir = tempfile.mkdtemp(prefix="optbench_")
    try:
        summary = tester.benchmark_once(candidate_cfg, steps=4, log_dir=tmpdir, seed=7)
        # JSON exists
        json_path = os.path.join(tmpdir, "trav_none_all.summary.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            payload = json.load(f)
        assert payload["candidate"]["optimizer"]["name"] == "trav_none_all"
        assert payload["baseline"]["optimizer"]["name"] == "AdamW"
        # TensorBoard event file exists
        tb_dir = payload["log_dir"]
        assert os.path.isdir(tb_dir)
        event_files = [p for p in os.listdir(tb_dir) if "events.out.tfevents" in p]
        assert len(event_files) >= 1
    finally:
        shutil.rmtree(tmpdir)

def test_run_suite_writes_index_file():
    train, val = tiny_dataset()
    tester = OptimizerComparisonTester(lambda: TinyNet(), train, val, eval_every=2)
    cfgs = [
        OptimizerConfig("trav_none_all", "v1.TraversalAdamW",
                        dict(traversal_mode="none", moment1_source="trav", moment2_source="trav", lr=1e-3)),
        OptimizerConfig("trav_full_mixed", "v1.TraversalAdamW",
                        dict(traversal_mode="full", moment1_source="grad", moment2_source="trav", lr=1e-3)),
    ]
    tmpdir = tempfile.mkdtemp(prefix="optbench_suite_")
    try:
        results = tester.run_suite(cfgs, steps=3, log_dir=tmpdir, seed=0)
        assert os.path.exists(os.path.join(tmpdir, "suite_results.json"))
        assert len(results) == 2
        # sanity: both runs are recorded
        names = {r["run_name"] for r in results}
        assert "trav_none_all" in names and "trav_full_mixed" in names
    finally:
        shutil.rmtree(tmpdir)