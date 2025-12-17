from core.learner.learner import Learner
from core.optim.feature_upgd.first_order import FirstOrderNonprotectingLocalUPGD, FirstOrderLocalUPGD, FirstOrderNonprotectingGlobalUPGD, FirstOrderGlobalUPGD

# Second-order imports require HesScale which has compatibility issues with PyTorch 2.x
try:
    from core.optim.feature_upgd.second_order import SecondOrderNonprotectingLocalUPGD, SecondOrderLocalUPGD, SecondOrderNonprotectingGlobalUPGD, SecondOrderGlobalUPGD
    FEATURE_SECOND_ORDER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Feature second-order UPGD not available: {e}")
    SecondOrderNonprotectingLocalUPGD = None
    SecondOrderLocalUPGD = None
    SecondOrderNonprotectingGlobalUPGD = None
    SecondOrderGlobalUPGD = None
    FEATURE_SECOND_ORDER_AVAILABLE = False

class FeatureFirstOrderLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderLocalUPGD
        name = "feature_upgd_fo_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureFirstOrderNonprotectingLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderNonprotectingLocalUPGD
        name = "feature_upgd_nonprotecting_fo_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureFirstOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderGlobalUPGD
        name = "feature_upgd_fo_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureFirstOrderNonprotectingGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderNonprotectingGlobalUPGD
        name = "feature_upgd_nonprotecting_fo_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSecondOrderLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderLocalUPGD
        name = "feature_upgd_so_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSecondOrderNonprotectingLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderNonprotectingLocalUPGD
        name = "feature_upgd_nonprotecting_so_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSecondOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderGlobalUPGD
        name = "feature_upgd_so_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSecondOrderNonprotectingGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderNonprotectingGlobalUPGD
        name = "feature_upgd_nonprotecting_so_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)