from core.learner.learner import Learner
from core.optim.weight_upgd.first_order import FirstOrderLocalUPGD, FirstOrderNonprotectingLocalUPGD, FirstOrderGlobalUPGD, FirstOrderNonprotectingGlobalUPGD
from core.optim.weight_upgd.first_order_clamped import FirstOrderGlobalUPGDClamped052
from core.optim.weight_upgd.first_order_layerselective import FirstOrderGlobalUPGDLayerSelective
from core.optim.weight_upgd.first_order_clamped_symmetric import FirstOrderGlobalUPGDClampedSymmetric

# Second-order imports require HesScale which has compatibility issues with PyTorch 2.x
# Make import optional to allow first-order learners to work
try:
    from core.optim.weight_upgd.second_order import SecondOrderLocalUPGD, SecondOrderNonprotectingLocalUPGD, SecondOrderGlobalUPGD, SecondOrderNonprotectingGlobalUPGD
    SECOND_ORDER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Second-order UPGD not available due to HesScale compatibility issue: {e}")
    SecondOrderLocalUPGD = None
    SecondOrderNonprotectingLocalUPGD = None
    SecondOrderGlobalUPGD = None
    SecondOrderNonprotectingGlobalUPGD = None
    SECOND_ORDER_AVAILABLE = False

class FirstOrderLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderLocalUPGD
        name = "upgd_fo_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderLocalUPGD
        name = "upgd_so_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderNonprotectingLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderNonprotectingLocalUPGD
        name = "upgd_nonprotecting_fo_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderNonprotectingLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderNonprotectingLocalUPGD
        name = "upgd_nonprotecting_so_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderGlobalUPGD
        name = "upgd_fo_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderGlobalUPGD
        name = "upgd_so_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderNonprotectingGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderNonprotectingGlobalUPGD
        name = "upgd_nonprotecting_fo_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderNonprotectingGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderNonprotectingGlobalUPGD
        name = "upgd_nonprotecting_so_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
class FirstOrderGlobalUPGDClamped052Learner(Learner):
    """
    First-order global UPGD with utilities clamped to max 0.52.
    
    Tests whether the ~0.1% of parameters with utilities > 0.52 are critical.
    If performance drops compared to standard UPGD, it proves the high-utility tail matters.
    """
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderGlobalUPGDClamped052
        name = "upgd_fo_global_clamped052"
        super().__init__(name, network, optimizer, optim_kwargs)

# Layer-selective gating learners
class FirstOrderGlobalUPGDLayerSelectiveLearner(Learner):
    """Base class for layer-selective UPGD. Subclasses specify gating_mode."""
    def __init__(self, network=None, optim_kwargs={}, gating_mode='full'):
        optim_kwargs = {**optim_kwargs, 'gating_mode': gating_mode}
        optimizer = FirstOrderGlobalUPGDLayerSelective
        name = f"upgd_fo_global_{gating_mode.replace('_', '')}"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDLayerSelectiveFullLearner(FirstOrderGlobalUPGDLayerSelectiveLearner):
    def __init__(self, network=None, optim_kwargs={}):
        super().__init__(network, optim_kwargs, gating_mode='full')

class UPGDLayerSelectiveOutputOnlyLearner(FirstOrderGlobalUPGDLayerSelectiveLearner):
    def __init__(self, network=None, optim_kwargs={}):
        super().__init__(network, optim_kwargs, gating_mode='output_only')

class UPGDLayerSelectiveHiddenOnlyLearner(FirstOrderGlobalUPGDLayerSelectiveLearner):
    def __init__(self, network=None, optim_kwargs={}):
        super().__init__(network, optim_kwargs, gating_mode='hidden_only')

class UPGDLayerSelectiveHiddenAndOutputLearner(FirstOrderGlobalUPGDLayerSelectiveLearner):
    def __init__(self, network=None, optim_kwargs={}):
        super().__init__(network, optim_kwargs, gating_mode='hidden_and_output')

# Symmetric clamping learners
class FirstOrderGlobalUPGDClampedSymmetricLearner(Learner):
    """Base class for symmetric clamping. Subclasses specify min/max clamp."""
    def __init__(self, network=None, optim_kwargs={}, min_clamp=0.48, max_clamp=0.52):
        optim_kwargs = {**optim_kwargs, 'min_clamp': min_clamp, 'max_clamp': max_clamp}
        optimizer = FirstOrderGlobalUPGDClampedSymmetric
        range_str = f"{int(min_clamp*100)}_{int(max_clamp*100)}"
        name = f"upgd_fo_global_clamped_{range_str}"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDClamped48_52Learner(FirstOrderGlobalUPGDClampedSymmetricLearner):
    """Clamp utilities to [0.48, 0.52] - very narrow."""
    def __init__(self, network=None, optim_kwargs={}):
        super().__init__(network, optim_kwargs, min_clamp=0.48, max_clamp=0.52)

class UPGDClamped44_56Learner(FirstOrderGlobalUPGDClampedSymmetricLearner):
    """Clamp utilities to [0.44, 0.56] - narrow."""
    def __init__(self, network=None, optim_kwargs={}):
        super().__init__(network, optim_kwargs, min_clamp=0.44, max_clamp=0.56)

class UPGDClamped40_60Learner(FirstOrderGlobalUPGDClampedSymmetricLearner):
    """Clamp utilities to [0.40, 0.60] - moderate."""
    def __init__(self, network=None, optim_kwargs={}):
        super().__init__(network, optim_kwargs, min_clamp=0.40, max_clamp=0.60)
