# tester.py
import os, json, time, copy, math, argparse, itertools, random
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Baseline
from torch.optim import AdamW

# Candidate optimizer (your variant lives in v1.py)
from v1 import TraversalAdamW

# ------------------------- small helpers -------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str) -> Tuple[float, float]:
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        b = y.size(0)
        tot_loss += float(loss) * b
        tot_acc += (logits.argmax(1) == y).float().sum().item()
        n += b
    return tot_loss / max(n, 1), tot_acc / max(n, 1)

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(1) == y).float().mean().item())

# ------------------------- dataclasses -------------------------

@dataclass
class OptimizerConfig:
    """Describes an optimizer instantiation."""
    name: str
    cls_path: str               # "v1.TraversalAdamW" or "torch.optim.AdamW"
    kwargs: Dict

    def as_readable(self) -> Dict:
        k = dict(sorted(self.kwargs.items()))
        return {"name": self.name, "class": self.cls_path, "kwargs": k}

# ------------------------- default models / data (optional) -------------------------

class SmallCNN(nn.Module):
    """No dropout/bn – reduces randomness across copies and keeps step-times tiny."""
    def __init__(self, in_ch: int = 1, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if in_ch == 1 else 64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)

def make_mnist_loaders(data_dir: str, batch_size: int = 128, subset: Optional[int] = None):
    import torchvision as tv
    import torchvision.transforms as T
    Ttrain = T.Compose([T.ToTensor()])
    Ttest  = T.Compose([T.ToTensor()])
    train = tv.datasets.MNIST(data_dir, train=True,  download=True, transform=Ttrain)
    test  = tv.datasets.MNIST(data_dir, train=False, download=True, transform=Ttest)
    if subset:
        idx = torch.randperm(len(train))[:subset]
        train = torch.utils.data.Subset(train, idx)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )

# ------------------------- the tester class -------------------------

class OptimizerComparisonTester:
    """
    Benchmarks a candidate optimizer configuration against a fixed AdamW baseline.
    - Logs per-step metrics to TensorBoard for both optimizers.
    - Uses two independent model copies starting from the same random initialization.
    - Saves a JSON summary at the end.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
        eval_every: int = 100,
    ):
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.eval_every = max(1, int(eval_every))

    # ---- factories ----------------------------------------------------------

    def _make_baseline(self, params, lr: float = 1e-3, weight_decay: float = 0.01,
                       betas=(0.9, 0.999), eps: float = 1e-8):
        """AdamW defaults commonly used as a simple baseline."""
        return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    def _make_candidate(self, params, cfg: OptimizerConfig):
        # Only TraversalAdamW is needed per request; hook left open for variations.
        return TraversalAdamW(params, **cfg.kwargs)

    # ---- core runner --------------------------------------------------------

    def benchmark_once(
        self,
        candidate_cfg: OptimizerConfig,
        steps: int,
        log_dir: str,
        seed: int = 123,
        baseline_kwargs: Optional[Dict] = None,
        run_name: Optional[str] = None,
    ) -> Dict:
        """
        Runs a single benchmark: candidate vs baseline from the *same init*.
        Returns a summary dict and writes TensorBoard logs + JSON.
        """
        os.makedirs(log_dir, exist_ok=True)
        run_name = run_name or f"{candidate_cfg.name}"
        writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

        set_seed(seed)

        # Make one init, then deep-copy for fairness
        base_model = self.model_fn()
        cand_model = copy.deepcopy(base_model)

        base_model.to(self.device)
        cand_model.to(self.device)

        baseline_kwargs = baseline_kwargs or {}
        base_opt = self._make_baseline(base_model.parameters(), **baseline_kwargs)
        cand_opt = self._make_candidate(cand_model.parameters(), candidate_cfg)

        train_iter = itertools.cycle(self.train_loader)
        t0 = time.time()
        base_model.train()
        cand_model.train()

        # log initial validation if available
        step0_metrics = {}
        if self.val_loader is not None:
            bvl, bva = evaluate(base_model, self.val_loader, self.loss_fn, self.device)
            cvl, cva = evaluate(cand_model, self.val_loader, self.loss_fn, self.device)
            writer.add_scalar("baseline/val_loss", bvl, 0); writer.add_scalar("baseline/val_acc", bva, 0)
            writer.add_scalar("candidate/val_loss", cvl, 0); writer.add_scalar("candidate/val_acc", cva, 0)
            step0_metrics = {"baseline": {"val_loss": bvl, "val_acc": bva},
                             "candidate": {"val_loss": cvl, "val_acc": cva}}

        # step loop
        for step in range(1, steps + 1):
            (xb, yb) = next(train_iter)
            xb, yb = xb.to(self.device), yb.to(self.device)

            # ---- baseline step
            base_opt.zero_grad(set_to_none=True)
            logits_b = base_model(xb)
            loss_b = self.loss_fn(logits_b, yb)
            loss_b.backward()
            base_opt.step()

            # ---- candidate step (uses same batch)
            cand_opt.zero_grad(set_to_none=True)
            logits_c = cand_model(xb)
            loss_c = self.loss_fn(logits_c, yb)
            loss_c.backward()
            cand_opt.step()

            # per-step logging
            writer.add_scalar("baseline/train_loss", float(loss_b), step)
            writer.add_scalar("candidate/train_loss", float(loss_c), step)
            writer.add_scalar("baseline/train_acc", accuracy_from_logits(logits_b, yb), step)
            writer.add_scalar("candidate/train_acc", accuracy_from_logits(logits_c, yb), step)

            if self.val_loader is not None and (step % self.eval_every == 0 or step == steps):
                bvl, bva = evaluate(base_model, self.val_loader, self.loss_fn, self.device)
                cvl, cva = evaluate(cand_model, self.val_loader, self.loss_fn, self.device)
                writer.add_scalar("baseline/val_loss", bvl, step); writer.add_scalar("baseline/val_acc", bva, step)
                writer.add_scalar("candidate/val_loss", cvl, step); writer.add_scalar("candidate/val_acc", cva, step)

        wall = time.time() - t0
        writer.flush(); writer.close()

        # final eval
        final = {"baseline": {}, "candidate": {}}
        if self.val_loader is not None:
            bvl, bva = evaluate(base_model, self.val_loader, self.loss_fn, self.device)
            cvl, cva = evaluate(cand_model, self.val_loader, self.loss_fn, self.device)
            final["baseline"].update({"val_loss": bvl, "val_acc": bva})
            final["candidate"].update({"val_loss": cvl, "val_acc": cva})

        # pack summary
        summary = {
            "run_name": run_name,
            "seed": seed,
            "steps": steps,
            "device": self.device,
            "baseline": {
                "optimizer": {"name": "AdamW", "class": "torch.optim.AdamW",
                              "kwargs": dict(sorted((baseline_kwargs or {}).items()))},
            },
            "candidate": {"optimizer": candidate_cfg.as_readable()},
            "initial_eval": step0_metrics,
            "final_eval": final,
            "wall_time_sec": wall,
            "log_dir": os.path.abspath(os.path.join(log_dir, run_name)),
        }
        # JSON artifact beside tb logs
        with open(os.path.join(log_dir, f"{run_name}.summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ---- suite API ----------------------------------------------------------

    def run_suite(
        self,
        candidate_cfgs: List[OptimizerConfig],
        steps: int,
        log_dir: str,
        seed: int = 123,
        baseline_kwargs: Optional[Dict] = None,
    ) -> List[Dict]:
        results = []
        for cfg in candidate_cfgs:
            res = self.benchmark_once(
                candidate_cfg=cfg,
                steps=steps,
                log_dir=log_dir,
                seed=seed,
                baseline_kwargs=baseline_kwargs,
                run_name=cfg.name,
            )
            results.append(res)
        # also save a top-level index
        idx_path = os.path.join(log_dir, "suite_results.json")
        with open(idx_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

# ------------------------- convenience factory for simple MNIST run ----------

def build_default_mnist_tester(batch_size: int = 128, subset: Optional[int] = None) -> Tuple[OptimizerComparisonTester, str]:
    data_dir = os.path.expanduser("~/.pytorch-datasets")
    train, test = make_mnist_loaders(data_dir, batch_size=batch_size, subset=subset)
    model_fn = lambda: SmallCNN(in_ch=1, num_classes=10)
    tester = OptimizerComparisonTester(model_fn, train, test, eval_every=100)
    return tester, "mnist"