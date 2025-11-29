from core.learner.learner import Learner
from core.optim.sgd import SGD
from core.optim.sgd_curvature_gating import SGDCurvatureGating


class SGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SGD
        name = "sgd"
        super().__init__(name, network, optimizer, optim_kwargs)

class SGDLearnerWithHesScale(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SGD
        name = "sgd_with_hesscale"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class SGDCurvatureGatingLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        from core.optim.weight_upgd.input_aware import compute_input_curvature_finite_diff
        import torch

        optimizer = SGDCurvatureGating
        name = "sgd_curvature_gating"

        # Filter out run-specific parameters
        run_specific_params = {'compute_curvature_every', 'n_samples', 'task', 'learner', 'save_path', 'seed', 'network'}
        filtered_kwargs = {k: v for k, v in optim_kwargs.items() if k not in run_specific_params}

        # Convert string parameters to appropriate types
        if 'lr' in filtered_kwargs:
            filtered_kwargs['lr'] = float(filtered_kwargs['lr'])
        if 'weight_decay' in filtered_kwargs:
            filtered_kwargs['weight_decay'] = float(filtered_kwargs['weight_decay'])
        if 'beta_utility' in filtered_kwargs:
            filtered_kwargs['beta_utility'] = float(filtered_kwargs['beta_utility'])
        if 'sigma' in filtered_kwargs:
            filtered_kwargs['sigma'] = float(filtered_kwargs['sigma'])
        if 'curvature_threshold' in filtered_kwargs:
            filtered_kwargs['curvature_threshold'] = float(filtered_kwargs['curvature_threshold'])
        if 'curvature_scale' in filtered_kwargs:
            filtered_kwargs['curvature_scale'] = float(filtered_kwargs['curvature_scale'])
        if 'beta_curvature' in filtered_kwargs:
            filtered_kwargs['beta_curvature'] = float(filtered_kwargs['beta_curvature'])

        super().__init__(name, network, optimizer, filtered_kwargs)

        # Store hutchinson_samples for curvature computation (default to 10)
        self.hutchinson_samples = int(optim_kwargs.get('hutchinson_samples', 10))

    def compute_input_curvature(self, model, input_batch, targets, criterion, return_per_sample=False):
        """Compute input curvature using finite differences method."""
        from core.optim.weight_upgd.input_aware import compute_input_curvature_finite_diff

        per_sample_curvatures = compute_input_curvature_finite_diff(
            model=model,
            inputs=input_batch,
            targets=targets,
            criterion=criterion,
            h=1e-3,  # Perturbation size
            niter=self.hutchinson_samples,  # Number of random directions
            temp=1.0  # Temperature scaling
        )

        if return_per_sample:
            return per_sample_curvatures
        else:
            return per_sample_curvatures.mean()

    def update_optimizer_curvature(self, curvature):
        """Update the optimizer with current input curvature."""
        if hasattr(self, 'optimizer_instance'):
            self.optimizer_instance.set_input_curvature(curvature)