import re
import pandas as pd
import numpy as np

import random
import tensorflow as tf
from layers import Encoder, Decoder


class SymbolicModel(tf.keras.Model):
    """Initializing the main model that calls
        1. Encoder layer
        2. Decoder layer
        from layers.py and use the call method to run the model
    """

    def __init__(self, units, context_text_processor, target_text_processor, is_gru, num_heads):

        super().__init__()
      
        # encoder and decoder from the 
        encoder = Encoder(context_text_processor, units, is_gru)
        decoder = Decoder(target_text_processor, units, num_heads)

        self.encoder = encoder
        self.decoder = decoder


    def call(self, inputs):
        """Call method runs the inputs through the encoder and decoder layer to get the final prediction
        """
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        return logits


    def translate(self, input_symbols, *, max_length=40, temperature=0.0):
        """Translate the input sequence into an output sequence using the decoder method by
            1. embedding the input input mathematical expressions symbols into a vector,
            2. getting the first initial state given the context,
            3. generating output expression until dthe max_length or [END] token encountered, and
            4. returning the final tokens together

        source code: tensorflow seq2seq tutorial, with minor modification
        """

        # no. 1 step
        context = self.encoder.convert_input(input_symbols)
        batch_size = tf.shape(input_symbols)[0]

        # no. 2 step
        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)

        # no. 3 step
        for i in range(max_length):
            next_token, done, state = self.decoder.get_next_token(context, next_token, done,  state, temperature)
          
            # store current token into final output tokens
            tokens.append(next_token)
            attention_weights.append(self.decoder.last_attention_weights)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # no. 4 step
        tokens = tf.concat(tokens, axis=-1)
        self.last_attention_weights = tf.concat(attention_weights, axis=1)

        result = self.decoder.tokens_to_symbols(tokens)

        return result


class Export(tf.Module):
    """Class to export custom function translate

    source code: tensorflow seq2seq tutorial
    """

    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):

        return self.model.translate(inputs)

