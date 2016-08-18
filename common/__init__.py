from .utils import  check_and_throw_if_fail
from .backend import get_shape, unpack, inner_product, top_k, trim_right_padding
from .layers import  BiDirectionalLayer, MLPClassifierLayer, GRUCell, RNNDecoderLayer, AttentionLayer, RNNDecoderLayerWithBeamSearch
from .objectives import categorical_crossentropy_ex
