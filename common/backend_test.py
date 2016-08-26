'''
Created on Aug 4, 2016

@author: lxh5147
'''
import unittest
from .backend import get_shape, get_time_step_length_without_padding, get_k_best_from_lattice, unpack, gather_by_sample, reverse, reshape, inner_product, top_k, choose_by_cond, _beam_search_one_step
from keras.layers import Input
import keras.backend as K
import numpy as np
from keras.layers import Embedding
class BackendTest(unittest.TestCase):

    def test_get_shape(self):
        # Input is a keras tensor
        x = Input((3,))
        self.assertEqual(get_shape(x), (None, 3), 'get_shape')
        x = K.variable((None, 3))
        # Placeholder is keras tensor
        x = K.placeholder(shape = (None, 3), ndim = 2)
        self.assertEqual(get_shape(x), (None, 3), 'get_shape')
        # Variable is not a keras tensor
        x = K.variable(np.random.random((3, 2)))
        try:
            get_shape(x)
            self.assertTrue(False, 'should not be executed')
        except Exception as e:
            self.assertTrue(e.message.startswith('You tried to call get_shape on'), 'get_shape')
            self.assertTrue(True, 'get_shape for non keras tensor')

    def test_get_time_step_length_without_padding(self):
        x = Input((2, 3))
        length_without_padding = get_time_step_length_without_padding(x)
        f = K.function(inputs = [x], outputs = [length_without_padding])
        x_val = [[[1, 2, 3], [0, 0, 0]], [[1, 2, 3], [0, 0, 0]]]
        len_val = f([x_val])[0]
        self.assertTrue(np.array_equal(len_val, 1), "len_val")

    def test_unpack(self):
        x = Input((3,))
        # num must match the shape of data input; otherwise will get exception
        x_list = unpack(x, 2)
        f = K.function(inputs = [x], outputs = x_list)
        x_val = [[3, 2, 2], [4, 2, 2]]
        x_list_val = f([x_val])
        self.assertEqual(len(x_list_val), 2, "x_list_val")
        self.assertTrue(np.array_equal(x_list_val[0], [3, 2, 2]), "x_list_val")
        self.assertTrue(np.array_equal(x_list_val[1], [4, 2, 2]), "x_list_val")

    def test_reverse(self):
        x = Input((3,))
        reversed_x = reverse(x, 2)
        f = K.function(inputs = [x], outputs = [reversed_x])
        x_val = [[3, 2, 2], [4, 2, 2]]
        reversed_x_val = f([x_val])[0]
        self.assertTrue(np.array_equal(reversed_x_val, [ [4, 2, 2], [3, 2, 2]]), "reversed_x_val")

    def test_reshape(self):
        x = Input((4,))
        y = reshape(x, shape = (-1, 2, 2))  # - means all the remaining
        f = K.function(inputs = [x], outputs = [y])
        x_val = [[3, 2, 2, 4], [4, 2, 2, 4]]
        y_val = f([x_val])[0]
        self.assertTrue(np.array_equal(y_val, [[ [3, 2], [2, 4]], [[4, 2], [2, 4]]]), "y_val")

    def test_inner_product(self):
        x = K.placeholder(shape = (3,))
        y = K.placeholder(shape = (3,))
        inner_prod = inner_product(x, y)
        f = K.function(inputs = [x, y], outputs = [inner_prod])
        inner_prod_val = f ([[2, 3, 4], [3, 4, 5]])[0]
        self.assertEqual(6 + 12 + 20, inner_prod_val, "inner_prod_val")
        # more complicated inner product
        x = Input((3,))
        y = K.placeholder(shape = (3,))
        inner_prod = inner_product(x, y)
        f = K.function(inputs = [x, y], outputs = [inner_prod])
        inner_prod_val = f ([[[1, 2, 1], [2, 1, 2]], [1, 2, 3]])[0]
        self.assertTrue(np.array_equal(inner_prod_val, [1 + 4 + 3, 2 + 2 + 6]), "inner_prod_val")

    def test_top_k(self):
        x = K.placeholder(shape = (6,))
        x_top_k, indices_top_k = top_k(x, k = 2)
        f = K.function(inputs = [x], outputs = [x_top_k, indices_top_k])
        x_top_k_val, indices_top_k_val = f ([[2, 3, 4, 5, 1, 6]])
        self.assertTrue(np.array_equal(x_top_k_val, [6, 5]), "x_top_k_val")
        self.assertTrue(np.array_equal(indices_top_k_val, [5, 3]), "indices_top_k_val")
        # more complicated
        x = K.placeholder(shape = (2, 3))
        x_top_k, indices_top_k = top_k(x, k = 2)
        f = K.function(inputs = [x], outputs = [x_top_k, indices_top_k])
        x_top_k_val, indices_top_k_val = f ([[[2, 3, 4], [5, 1, 6]]])
        self.assertTrue(np.array_equal(x_top_k_val, [[4, 3], [6, 5]]), "x_top_k_val")
        self.assertTrue(np.array_equal(indices_top_k_val, [[2, 1], [2, 0]]), "indices_top_k_val")

    def test_choose_with_cond(self):
        x1 = K.placeholder(shape = (3,))
        x2 = K.placeholder(shape = (3,))
        cond = K.placeholder(shape = (3,), dtype = "int32")
        x_selected = choose_by_cond(cond, x1, x2)
        f = K.function(inputs = [cond, x1, x2], outputs = [x_selected])
        x_selected_val = f ([[0, 0, 1], [1, 2, 3], [4, 5, 6] ])[0]
        self.assertTrue(np.array_equal(x_selected_val, [4, 5, 3]), "x_selected_val")


    def test_gather_by_sample(self):
        x = K.placeholder(shape = (2, 3))
        indices = K.placeholder(shape = (2,), dtype = 'int32')
        x_selected = gather_by_sample(x, indices)
        f = K.function(inputs = [x, indices], outputs = [x_selected])
        x_selected_val = f ([[[1, 2, 3], [4, 5, 6]], [2, 1]])[0]
        self.assertTrue(np.array_equal(x_selected_val, [3, 5]), "x_selected_val")
        # more complicated case
        x = K.placeholder(shape = (2, 2, 2))
        indices = K.placeholder(shape = (2,), dtype = 'int32')
        x_selected = gather_by_sample(x, indices)
        f = K.function(inputs = [x, indices], outputs = [x_selected])
        x_selected_val = f ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [0, 1]])[0]
        self.assertTrue(np.array_equal(x_selected_val, [[1, 2], [7, 8]]), "x_selected_val")

    def test_beam_search_one_step(self):
        # _step_score, _state, output_score, number_of_samples, beam_size, state_dim, output_score_list, prev_output_index_list, output_label_id_list, embedding
        nb_samples = 2
        beam_size = 3
        state_dim = 2
        output_dim = 4
        embedding_dim = 2

        _step_score = K.placeholder(shape = (nb_samples * beam_size, output_dim))
        _state = K.placeholder(shape = (nb_samples * beam_size, state_dim))
        output_score = K.placeholder(shape = (nb_samples * beam_size,))
        embedding_weights = np.array([[0.0, 0.0], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        embedding = Embedding(output_dim, embedding_dim, weights = [embedding_weights])
        output_score_list = []
        prev_output_index_list = []
        output_label_id_list = []
        _tensors_to_debug = []
        updated_output_score, current_input, current_state = _beam_search_one_step(_step_score, _state, output_score, nb_samples, beam_size, state_dim, output_score_list, prev_output_index_list, output_label_id_list, embedding, _tensors_to_debug)
        f = K.function(inputs = [_step_score, _state, output_score], outputs = [output_score_list[-1], prev_output_index_list[-1], output_label_id_list[-1], updated_output_score, current_input, current_state] + _tensors_to_debug)
        _step_score_val = [ [0.1, 0.2, 0.3, 0.4],
                            [0.2, 0.1, 0.4, 0.3],
                            [0.3, 0.1, 0.2, 0.4],  # 1st sample
                            [0.1, 0.15, 0.25, 0.5],
                            [0.15, 0.1, 0.25, 0.5],
                            [0.25, 0.15, 0.5, 0.1],  # 2nd sample
                           ]  # 2*3,4
        _state_val = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4],  # 1st sample
                  [0.5, 0.6], [0.6, 0.7], [0.7, 0.8],  # 2nd sample
                           ]  # 2*3,2
        output_score_val = [-0.01, -0.02, -0.03, -0.015, -0.025, -0.035 ]  # 2*3

        current_output_score_val, prev_output_index_val, current_output_label_id_val, updated_output_score_val, current_input_val, current_state_val, _score_val, _top_score_val, _top_indice_val = f([_step_score_val, _state_val, output_score_val])
        # _score_val: nb_samples, beam_size*output_dim
        _score_val_ref = np.reshape(np.expand_dims(output_score_val, 1) + np.log (_step_score_val), (nb_samples , beam_size * output_dim))
        self.assertTrue(np.sum(np.abs(_score_val - _score_val_ref)) < 0.001, '_score_val')

        sort_indices = np.argsort(_score_val, axis = 1)
        _top_indice_val_ref = sort_indices[:, -1:-beam_size - 1:-1]
        self.assertTrue(np.array_equal(_top_indice_val, _top_indice_val_ref), "_top_indice_val")

        prev_output_index_val_ref = _top_indice_val_ref // output_dim
        self.assertTrue(np.array_equal(prev_output_index_val, prev_output_index_val_ref), "prev_output_index_val")
        current_output_label_id_val_ref = _top_indice_val_ref - prev_output_index_val_ref * output_dim  # this assume that output dim is greater than beam_size
        self.assertTrue(np.array_equal(current_output_label_id_val, current_output_label_id_val_ref), "prev_output_index_val")

        _score_val_sorted = np.sort(_score_val, axis = 1)
        _top_score_val_ref = _score_val_sorted[:, -1:-beam_size - 1:-1]
        self.assertTrue(np.sum(np.abs(_top_score_val - _top_score_val_ref)) < 0.001, '_top_score_val')

        current_input_val_ref = embedding_weights[np.reshape(current_output_label_id_val, newshape = (-1,))]
        self.assertTrue(np.sum(np.abs(current_input_val - current_input_val_ref)) < 0.001, "current_input_val")

        self.assertTrue(np.sum(np.abs(current_output_score_val - _top_score_val)) < 0.001, "current_output_score_val")

        self.assertTrue(np.sum(np.abs(updated_output_score_val - np.reshape(_top_score_val, (-1,)))) < 0.001, "updated_output_score_val")

        current_state_val_ref = [ _state_val[i * beam_size + prev_output_index_val[i, j]] for i in range(nb_samples)  for j in range(beam_size)  ]
        self.assertTrue(np.sum(np.abs(current_state_val - np.array(current_state_val_ref))) < 0.001, "updated_output_score_val")

    def test_get_k_best_from_lattice(self):
        nb_samples = 2
        beam_size = 3
        time_steps = 2
        _tensors_to_debug = []
        output_label_id_list = [K.placeholder(shape = (nb_samples, beam_size), dtype = 'int32') for _ in range(time_steps)]
        prev_output_index_list = [K.placeholder(shape = (nb_samples, beam_size), dtype = 'int32') for _ in range(time_steps)]
        output_score_list = [K.placeholder(shape = (nb_samples, beam_size)) for _ in range(time_steps)]
        lattice = (K.pack(output_label_id_list), K.pack(prev_output_index_list), K.pack(output_score_list))
        output, output_score = get_k_best_from_lattice(lattice, k = 2, eos = -1, _tensors_to_debug = _tensors_to_debug)
        f = K.function(inputs = output_label_id_list + prev_output_index_list + output_score_list, outputs = [output, output_score] + _tensors_to_debug)
        output_label_id_list_val = [[[3, 2, 1], [1, 3, -1]] , [[2, 1, 3], [3, -1, -1]]]
        prev_output_index_list_val = [[[0, 0, 0], [0, 0, 0]] , [[0, 1, 2], [2, 2, 1]]]
        output_score_list_val = [[[-0.1, -0.2, -0.3], [-0.25, -0.36, -0.45]], [[-0.6, -0.5, -0.7], [-0.9, -1.2, -0.75]]]
        output_0 = [[2, 1], [3, 2]]
        output_1 = [[-1, 3], [3, -1]]
        output_val_ref = [output_0, output_1]  # nb_samples, k, time_steps
        output_score_val_ref = [[-0.5, -0.6], [-0.45, -0.75]]
        outputs_val = f(output_label_id_list_val + prev_output_index_list_val + output_score_list_val)
        output_val, output_score_val = outputs_val[:2]
        self.assertTrue(np.sum(np.abs(output_score_val - output_score_val_ref)) < 0.001, "output_score_val")
        self.assertTrue(np.array_equal(output_val, output_val_ref), "output_val")
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
