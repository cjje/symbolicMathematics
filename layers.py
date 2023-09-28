import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    """Encoder class has
        1. embedding layer for context sequence representation, and
        2. bidirectional rnn layer (either GRU or LSTM) to process the context representation
    """

    def __init__(self, input_context_processor, units, is_gru=True): 
        
        super(Encoder, self).__init__()
        self.input_context_processor = input_context_processor
        self.vocab_size = len(input_context_processor.get_vocabulary())
        self.units = units

        # embedding layer: context token representation
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        self.is_gru = is_gru

        # rnn layer: bidirectional GRU or LSTM layer
        if self.is_gru:
            self.rnn = tf.keras.layers.Bidirectional(
                merge_mode='sum',
                layer=tf.keras.layers.GRU(units, return_sequences=True) # recurrent_initializer='glorot_uniform')
            )
        else:
            self.rnn = tf.keras.layers.Bidirectional(
                merge_mode='sum',
                layer=tf.keras.layers.LSTM(units, return_sequences=True)
             )
                
    def call(self, x):

        """Call method
            1. takes the context and
            2. runs the embedded context in rnn layer and returns the output
        """

        # no 1. step
        x = self.embedding(x)

        # no. 2 step
        x = self.rnn(x)

        return x

    @tf.function
    def convert_input(self, symbols):
        """Convert input context to tokens
        source: tensorflow seq2seq tutorial
        """
        
        symbols = tf.convert_to_tensor(symbols)

        if len(symbols.shape) == 0:
            symbols = tf.convert_to_tensor(symbols)[tf.newaxis]
        context = tf.convert_to_tensor(self.input_context_processor(symbols))
        context = self(context)

        return context


class CrossAttention(tf.keras.layers.Layer):
    
    """ Attention class has
        1. MultiHeadAttention layer to allow selective attention to the relevants inputs from encoder,
        2. Add layer to add query output with the decoder output, and
        3. LayerNormalization layer to normalize the output from add layer.
    """

    def __init__(self, units, num_heads, **kwargs):
        
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=num_heads, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        """Call method 
            1. takes the decoder and context inputs
            2. computes a score vector, and
            3. add the decoder output and the score output
        """

        # no. 1 step
        attn_output, attn_scores = self.mha(query=x, value=context, return_attention_scores=True)

        # no. 2 step
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        self.last_attention_weights = attn_scores

        # no. 3 step
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class Decoder(tf.keras.layers.Layer):
    """Decoder class has
        1. embedding layer for target sequence representation,
        2. RNN layer (GRU) to keep track of the history,
        3. attention layer to query the result from the encoder layer,
        4. fully connected dense layer to process the final output tokens, and 
        5. stringlookup layer to convert symbols to token ids, and token ids to symbols

    """

    def __init__(self, output_target_processor, units, num_heads=4):

        super(Decoder, self).__init__()
        self.output_target_processor = output_target_processor
        self.vocab_size = len(output_target_processor.get_vocabulary())
        self.word_to_id = tf.keras.layers.experimental.preprocessing.StringLookup(
                                                       vocabulary=output_target_processor.get_vocabulary(),
                                                       mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.experimental.preprocessing.StringLookup(
                                                       vocabulary=output_target_processor.get_vocabulary(),
                                                       mask_token='', oov_token='[UNK]',
                                                       invert=True)
        self.start_token = self.word_to_id(tf.constant('[START]'))
        self.end_token = self.word_to_id(tf.constant('[END]'))
        self.units = units

        # layers 
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        self.rnn = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.attention = CrossAttention(units, num_heads=num_heads)
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation = 'tanh')


    def call(self, context, x, state=None, return_state=False):
        """Call method
            1. creates a target sequence representation,
            2. process the target sequences through an RNN layer,
            3. runs it through the attention layer, and
            4. returns the predictions for the next token using the dense layer
        """

        # no. 1 step
        x = self.embedding(x)
        
        # no. 2 step
        x, state = self.rnn(x, initial_state=state)
        
        # no. 3 step
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights

        # no. 4 step
        preds = self.dense(x)

        if return_state:
            return preds, state
        else:
            return preds


    def get_initial_state(self, context):
        """
        Returns
            1. the start_token ([START]),
            2. done variable (tracks if the sequence is done producing), and
            3. the initial state of the output given the embedded context input

        source code: tensorflow seq2seq tutorial

        """
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)

        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    
    def tokens_to_symbols(self, tokens):
        """
        Converts the token back to original input symbols

        source code: tensorflow seq2seq tutorial
        """
        symbols = self.id_to_word(tokens)
        result = tf.strings.reduce_join(symbols, axis=-1, separator=' ')

        return result

    
    def get_next_token(self, context, next_token, done, state, temperature = 0.0):
        """
        Returns
            1. next token given its knowledge about the current context, done, state (passed inputs)
            2. done state based on the passed inputs (done if the next token is '[END]')
            3. state based on the passed inputs

        source code: tensorflow seq2seq tutorial
        """
        logits, state = self(context, next_token, state = state, return_state=True) 

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :]/temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        done = (next_token == self.end_token)
        # once done, no more symbols are generated
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state    

