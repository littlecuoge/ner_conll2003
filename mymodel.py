import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

from utils import shared, set_values, get_name
from nn import *
from optimization import Optimization

#
# def inspect_inputs(i, node, fn):
#     print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs], end=' ')
#
#
# def inspect_outputs(i, node, fn):
#     print(" output(s) value(s):", [output[0] for output in fn.outputs])

class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            if len(self.name)>18:
                self.name = self.name[30:]
            # Model location
            model_path = os.path.join(models_path, self.name)
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                cPickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_word, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        # with open(self.mappings_path, 'wb') as f:
        #     mappings = {
        #         'id_to_word': self.id_to_word,
        #         'id_to_char': self.id_to_char,
        #         'id_to_tag': self.id_to_tag,
        #     }
        #     cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
              multibi,
              droplayer,
              conv,
              bd_style,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)

        # Number of capitalization features
        if cap_dim:
            n_cap = 4

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.imatrix(name='word_ids')
        char_for_ids = T.itensor3(name='char_for_ids')
        tag_ids = T.imatrix(name='tag_ids')
        s_lens = T.ivector(name='s_lens')
        word_ids2 = T.ivector()
        char_for_ids2 = T.imatrix()
        char_mask = T.ftensor3('char_mask')
        char_test_mask = T.fmatrix('char_test_mask')

        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')

        # Sentence length
        s_len = word_ids2.shape[-1]

        # Final input (all word features)
        input_dim = 0
        inputs = []
        test_input = []
        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            word_test_input = word_layer.link(word_ids2)
            inputs.append(word_input)
            test_input.append(word_test_input)
            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print('Loading pretrained embeddings from %s...' % pre_emb)
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print('WARNING: %i invalid lines' % emb_invalid)
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in xrange(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1
                word_layer.embeddings.set_value(new_weights)
                print('Loaded %i pretrained embeddings.' % len(pretrained))
                print ('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros, n_words,
                            100. * (c_found + c_lower + c_zeros) / n_words
                      )
                print ('%i found directly, %i after lowercasing, '
                       '%i after lowercasing + zero.') % (
                          c_found, c_lower, c_zeros
                      )

        #
        # Chars inputs
        #
        if char_dim:
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')
            char_convlayer = Convolution1d(char_dim, 3, name='char_convolution_layer')  # concolution window length
            char_for_output = char_convlayer.link(char_layer.link(char_for_ids))
            char_for_test_output = char_convlayer.link(char_layer.link(char_for_ids2.dimshuffle('x', 0, 1)))
            char_for_output = T.switch(char_for_output>0, char_for_output, 0)
            char_for_test_output = T.switch(char_for_test_output>0, char_for_test_output, 0)#mask implement
            char_for_output = char_for_output*char_mask.dimshuffle(0,1,2,'x')
            char_for_test_output = char_for_test_output * char_test_mask.dimshuffle('x', 0, 1, 'x')

            char_for_output = T.max(char_for_output,axis=2)
            char_for_test_output = T.max(char_for_test_output,axis=2)

            char_for_output = char_for_output.flatten(3)
            input_dim += char_dim
            char_for_test_output = char_for_test_output.flatten(3)

            inputs.append(char_for_output)
            test_input.append(char_for_test_output[0])

        # Prepare final input
        if len(inputs) != 1:
            inputs = T.concatenate(inputs, axis=2)

        test_input = T.concatenate(test_input, axis=1)
        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)
            test_input = (1 - dropout) * test_input

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=True, name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=True, name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[:, ::-1, :])
        word_for_output = word_lstm_for.h.dimshuffle(1, 0, 2)
        word_rev_output = word_lstm_rev.h[:, ::-1, :].dimshuffle(1,0,2)
        test_input = test_input.dimshuffle('x', 0, 1)
        word_lstm_for.link(test_input)
        word_lstm_rev.link(test_input[:, ::-1, :])
        final_output2 = T.concatenate([word_lstm_for.h.dimshuffle(1, 0, 2), word_lstm_rev.h[:, ::-1, :].dimshuffle(1, 0, 2)], axis=2)
        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=2
            )
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
            final_output2 = tanh_layer.link(final_output2)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer', activation=(None if crf else 'sigmoid'))

        tags_scores = final_layer.link(final_output)
        tags_scores2 = final_layer.link(final_output2)[0]

        if dropout and droplayer >= 3:
            dropout_layer3 = DropoutLayer(p=dropout)
            input_train = dropout_layer3.link(tags_scores)
            input_test = (1 - dropout) * tags_scores
            tags_scores = T.switch(T.neq(is_train, 0), input_train, input_test)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')
            def calulate_cost(tags_score, tag_id, s_len):
                small = -1000
                b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
                e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
                tags_score = tags_score[:s_len, :]
                tag_id = tag_id[:s_len]
                observations = T.concatenate(
                    [tags_score, small * T.ones((s_len, 2))],
                    axis=1
                )
                observations = T.concatenate(
                    [b_s, observations, e_s],
                    axis=0
                )
                # Score from tags
                real_path_score = tags_score[T.arange(s_len), tag_id].sum()
                # Score from transitions
                b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
                e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
                padded_tags_ids = T.concatenate([b_id, tag_id, e_id], axis=0)
                real_path_score += transitions[
                    padded_tags_ids[T.arange(s_len + 1)],
                    padded_tags_ids[T.arange(s_len + 1) + 1]
                ].sum()

                all_paths_scores = forward(observations, transitions)
                tmpcost = - (real_path_score - all_paths_scores)
                # calulate_cost = theano.function(inputs=[tags_score, tag_id],outputs=tmpcost)
                return tmpcost
            all_cost, _ = theano.scan(fn=calulate_cost, sequences=[tags_scores, tag_ids, s_lens],name='crf_scan')
            cost = all_cost.mean()

        small = -1000
        b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
        e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)

        observations = T.concatenate(
            [tags_scores2, small * T.ones((s_len, 2))],
            axis=1
        )
        observations = T.concatenate(
            [b_s, observations, e_s],
            axis=0
        )

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_convlayer)
            params.extend(char_layer.params)
            params.extend(char_convlayer.params)

        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)

        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)


        # Prepare train and eval inputs
        eval_inputs = []
        eval_inputs.append(word_ids)
        eval_inputs.append(char_for_ids)
        eval_inputs.append(s_lens)
        eval_inputs.append(char_mask)

        eval_inputs2 = []
        eval_inputs2.append(word_ids2)
        eval_inputs2.append(char_for_ids2)
        eval_inputs2.append(char_test_mask)
        # eval_inputs2.append(charpos)
            # if char_bidirect:
            #     eval_inputs.append(char_rev_ids)
            # eval_inputs.append(char_pos_ids)

        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print('Compiling...')
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})

            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs2,
                outputs=forward(observations, transitions, viterbi=True, return_alpha=False, return_best_sequence=True)
                # givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )

        return f_train, f_eval
