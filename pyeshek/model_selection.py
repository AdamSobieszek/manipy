import os
import torch
import logging
import numpy as np
import torch.nn as nn

# --- REVISED: Bayesian Pareto Model Selector ---
class BayesianParetoSelector:
    """
    Selects Pareto-optimal models using a fully probabilistic, Bayesian approach.

    This class encapsulates all statistical logic, including bootstrapping and
    metric calculation, making the Trainer's job simpler.
    """
    def __init__(self, metrics_to_run: dict, save_dir: str, n_bootstraps: int):
        """
        Initializes the selector.

        Args:
            metrics_to_run (dict): A dictionary mapping metric names to callable
                functions. e.g., {'logit_corr': compute_logit_corr}.
                The functions should accept (pred_logit, targ_logit, weights).
            save_dir (str): Directory to save model files for the Pareto front.
            n_bootstraps (int): The number of bootstrap samples to generate.
        """
        self.metrics_to_run = metrics_to_run
        self.directions = {k: ('max' if 'corr' in k else 'min') for k in metrics_to_run}
        self.save_dir = save_dir
        self.n_bootstraps = n_bootstraps
        self.pareto_front = [] # List of {'dists': {...}, 'file': '...'}
        self.last_run_dists = None # For logging and checkpointing

        os.makedirs(self.save_dir, exist_ok=True)
        logging.info(f"BayesianParetoSelector initialized. Bootstraps: {n_bootstraps}. Metrics: {list(self.metrics_to_run.keys())}")

    def _bootstrap_and_get_dists(self, model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, w_val: torch.Tensor) -> dict:
        """
        Internal method to perform bootstrapping and calculate metric distributions.
        This is the core statistical engine of the selector.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(x_val)
            pred_mean = model.get_mean(outputs)
            targ_mean = torch.nanmean(y_val, dim=1, keepdim=True)

            pred_logit = torch.logit(pred_mean.clamp(1e-8, 1 - 1e-8)).cpu().numpy().flatten()
            targ_logit = torch.logit(targ_mean.clamp(1e-8, 1 - 1e-8)).cpu().numpy().flatten()
            weights = w_val.cpu().numpy().flatten()

        n_val = len(x_val)
        rng = np.random.default_rng()
        
        # Initialize dictionary to hold bootstrap distributions for each metric
        boot_dists = {name: np.empty(self.n_bootstraps) for name in self.metrics_to_run}

        for i in range(self.n_bootstraps):
            indices = rng.integers(0, n_val, n_val)
            pred_b, targ_b, w_b = pred_logit[indices], targ_logit[indices], weights[indices]
            
            # Calculate every metric on the same bootstrap sample
            for name, func in self.metrics_to_run.items():
                boot_dists[name][i] = func(pred_b, targ_b, w_b)
        
        return boot_dists

    def _get_dominance_probabilities(self, dists_A: dict, dists_B: dict) -> tuple[float, float]:
        """Calculates (P(A dom B), P(B dom A)) from joint bootstrap distributions."""
        # This method is now fully self-contained and unchanged in logic
        metrics = list(self.directions.keys())
        vectors_A = np.stack([dists_A[m] for m in metrics], axis=1)
        vectors_B = np.stack([dists_B[m] for m in metrics], axis=1)
        
        rng = np.random.default_rng()
        indices_B = rng.integers(0, len(vectors_B), len(vectors_A))
        diff_vectors = vectors_A - vectors_B[indices_B]

        is_A_dominant = np.ones(len(diff_vectors), dtype=bool)
        for i, metric in enumerate(metrics):
            is_A_dominant &= (diff_vectors[:, i] > 0) if self.directions[metric] == 'max' else (diff_vectors[:, i] < 0)
        p_A_dom_B = np.mean(is_A_dominant)

        is_B_dominant = np.ones(len(diff_vectors), dtype=bool)
        for i, metric in enumerate(metrics):
            is_B_dominant &= (diff_vectors[:, i] < 0) if self.directions[metric] == 'max' else (diff_vectors[:, i] > 0)
        p_B_dom_A = np.mean(is_B_dominant)

        return p_A_dom_B, p_B_dom_A

    def step(self, model: nn.Module, val_data: tuple, epoch: int) -> bool:
        """
        Evaluates a model, updates the Pareto front, and returns if the model was added.
        """
        x_val, y_val, w_val = val_data
        
        # Step 1: Compute metric distributions for the candidate model
        candidate_dists = self._bootstrap_and_get_dists(model, x_val, y_val, w_val)
        self.last_run_dists = candidate_dists # Store for logging/checkpointing

        # Step 2: Compare with the existing Pareto front
        for member in self.pareto_front:
            _, prob_member_dominates = self._get_dominance_probabilities(candidate_dists, member['dists'])
            if prob_member_dominates > 0.5:
                logging.debug(f"Epoch {epoch}: Not added. Dominated by {os.path.basename(member['file'])} with P={prob_member_dominates:.3f}")
                return False

        new_front = []
        for member in self.pareto_front:
            prob_cand_dominates, _ = self._get_dominance_probabilities(candidate_dists, member['dists'])
            if prob_cand_dominates <= 0.5:
                new_front.append(member)
            else:
                logging.info(f"Epoch {epoch}: Removing {os.path.basename(member['file'])} (dominated with P={prob_cand_dominates:.3f})")
                try: os.remove(member['file'])
                except OSError as e: logging.error(f"Error removing file: {e}")

        save_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        new_front.append({'dists': candidate_dists, 'file': save_path})
        self.pareto_front = new_front
        logging.info(f"Epoch {epoch}: Model added to Pareto front. New front size: {len(self.pareto_front)}")
        return True

    def get_best_model_path(self, primary_metric: str) -> str or None:
        if not self.pareto_front: return None
        direction = self.directions.get(primary_metric)
        key_func = lambda x: np.mean(x['dists'][primary_metric])
        best_member = max(self.pareto_front, key=key_func) if direction == 'max' else min(self.pareto_front, key=key_func)
        logging.info(f"Final best model from Pareto front via '{primary_metric}': {best_member['file']}")
        return best_member['file']