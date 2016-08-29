from .utils import  check_and_throw_if_fail
from .backend import get_shape, unpack, inner_product, top_k, beam_search, get_k_best_from_lattice
from .layers import BiDirectionalLayer, MLPClassifierLayer, RNNDecoderLayer, AttentionLayer, RNNDecoderLayerWithBeamSearch
from .objectives import categorical_crossentropy_ex
