from .utils import  check_and_throw_if_fail
from .backend import get_k_best_from_lattice
from .layers import BiDirectionalLayer, MLPClassifierLayer, RNNDecoderLayer, AttentionLayer, RNNDecoderLayerWithBeamSearch, TimeDistributed
from .objectives import categorical_crossentropy_ex
from .training import convert_to_model_with_parallel_training
