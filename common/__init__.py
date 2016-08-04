from .utils import  check_and_throw_if_fail
from .backend import shape, unpack, inner_product, top_k, trim_right_padding
from .layers import  BiDirectionalLayer, MLPClassifierLayer, GRUCell, RNNLayer, AttentionLayer, RNNBeamSearchDecoder
from .objectives import categorical_crossentropy_ex
