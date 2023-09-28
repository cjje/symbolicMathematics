from layers import Encoder, CrossAttention, Decoder
from model import SymbolicModel
import tensorflow as tf
import pandas as pd
from model_params import model_params
import zipfile

class SymbolicModelTrain():


    def __init__(self, batch_size = 128):
        
        """
        train_ds: train dataset used in model.fit in tf.Dataset form
        valid_ds: validation dataset used in model.fit in tf.Dataset form
        batch_size: model batch size set at the beginning of the training
        model_params: reads from model_params.py where user can specify following hyperparameters to test
            1. epoch: number of training epoch
            2. optimizer: type of optimizers for model.fit (eg. adam, RMSProp)
            3. num_units: model dimension
            4. num_heads: number of attention heads for the CrossAttention (attention layer)
            5. is_gru: whether the Encoder layer uses GRU or LSTM for the RNN layer
        context_processor: tf TextVectorization layer to tokenize and clean input sequence
        target_processor: tf TextVectorization layer to tokenize and clean output sequence
        training_result: stores model and model run history by different hyperparameters 
        """


        self.train_ds = None
        self.valid_ds = None
        self.batch_size = batch_size
        
        self.model_params = model_params
        self.context_processor = None
        self.target_processor = None

        self.training_result = {}


    def load_data(self, data_path):
        """Load the data for symbolic integration task
        """

        df_train = pd.read_csv(f'{data_path}/integration.train', header=None, sep='\t',names=['X','y'])
        df_valid = pd.read_csv(f'{data_path}/integration.valid', header=None, sep='\t',names=['X','y'])

        # drop indices with NaN values from the data
        df_train.drop(index=df_train[df_train['X']!=df_train['X']].index, inplace=True)
        df_train.drop(index=df_train[df_train['y']!=df_train['y']].index, inplace=True)
        
        df_valid.drop(index=df_train[df_train['y']!=df_train['y']].index, inplace=True)
        df_valid.drop(index=df_train[df_train['y']!=df_train['y']].index, inplace=True)

        # Set training and validation data
        X_train = df_train['X'].values
        y_train = df_train['y'].values
        X_valid = df_valid['X'].values
        y_valid = df_valid['y'].values

        # dataset specific cleaning - exacting the actual expressions from the loaded data
        X_train = [i.split('|')[-1] for i in X_train]
        X_valid = [i.split('|')[-1] for i in X_valid]

        del df_train
        del df_valid

        print("X_train rows: ", len(X_train))
        print("y_train rows: ", len(y_train))
        print("X_valid rows: ", len(X_valid))
        print("y_valid rows: ", len(y_valid))

        # set intermediate train_intermed and valid_intermed
        self.train_intermed = (tf.data.Dataset
            .from_tensor_slices((X_train, y_train))
            .batch(self.batch_size)
            )

        self.valid_intermed = (tf.data.Dataset
            .from_tensor_slices((X_valid, y_valid))
            .batch(self.batch_size)
            )

        del X_train
        del X_valid
        del y_train
        del y_valid

    @tf.keras.utils.register_keras_serializable()
    def lower_and_strip_punctuation(self, input_text):

        """Process and clean the input tokens
        """

        # lower the input tokens
        input_text = tf.strings.lower(input_text)
        # keep space, a to z, and select punctuation.
        input_text = tf.strings.regex_replace(input_text, "[^ 0-9a-z.?!\'\-\+]", '')
        # strip potential white space
        input_text = tf.strings.strip(input_text)
        # add start and end symbols and split on white space
        input_text = tf.strings.join(['[START]', input_text, '[END]'], separator=' ')
        
        return input_text


    def set_processor(self):

        """Get context, target processors to extract tokens from context and target vectors
        """

        self.context_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize = self.lower_and_strip_punctuation,
            max_tokens=500)
        self.target_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize = self.lower_and_strip_punctuation,
            max_tokens=500)

        self.context_processor.adapt(self.train_intermed.map(lambda context, target: context))
        self.target_processor.adapt(self.valid_intermed.map(lambda context, target: target))


    def __process_text__(self, context, target):

        """ Convert the tensorflow dataset object to an appropriate form for model.fit
            source: tensorflow seq2seq tutorial
        """

        context = tf.convert_to_tensor(self.context_processor(context))
        target = self.target_processor(target)
        target_input = tf.convert_to_tensor(target[:,:-1])
        target_output = tf.convert_to_tensor(target[:,1:])

        return (context, target_input), target_output


    def set_train_datasets(self):

        """Run the batched intermed train and validation set through process_text method
        """

        self.train_ds = self.train_intermed.map(self.__process_text__, tf.data.AUTOTUNE)
        self.valid_ds = self.valid_intermed.map(self.__process_text__, tf.data.AUTOTUNE)

    @tf.function
    def masked_loss(self, y_true, y_pred):

        """ Custom loss function to ignore the padding in the output sequence
            source: tensorflow seq2seq tutorial
        """

        # loss for each item in the batch
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = loss_fn(y_true, y_pred)

        # ignore the losses on padding
        mask = tf.cast(y_true != 0, loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss)/tf.reduce_sum(mask)


    def train_model(self):

        """Train the model by iterating through a set of hyperparameters specified by the user in the model_params.py file
        """

        # run trials with different model hyper parameters
        for trial in self.model_params:

            print(f"Initializing {trial}...")
                
            trial_parm = self.model_params[trial]
            units = trial_parm['num_units']
            optimizer = trial_parm['optimizer']
            epoch = trial_parm['epoch']
            is_gru = trial_parm['is_gru']
            num_heads = trial_parm['num_heads']

            print(f"""Initializing {trial}:
                    - units: {units}
                    - optimizer: {optimizer}
                    - epoch: {epoch}
                    - is_gru: {is_gru}
                    - num_heads: {num_heads}
                    """)
            print("")


            model = SymbolicModel(units, self.context_processor, self.target_processor, is_gru, num_heads)
            model.compile(optimizer=optimizer,
                            loss = self.masked_loss,
                            metrics=['accuracy', self.masked_loss])
              
            history = model.fit(
                                  self.train_ds.repeat(),
                                  epochs=epoch,
                                  steps_per_epoch = 100,
                                  validation_data=self.valid_ds,
                                  validation_steps = 20,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
                                  )
                

            # store the trained model and the related parameters
            if trial not in self.training_result:
                self.training_result[trial] = {}
                        
            self.training_result[trial]['trained_model'] = model
            self.training_result[trial]['history'] = history

            print(f"Done running {trial} and saved the result")
            print("")
        

    def run_all(self, data_path="data"):

        """Run all training steps by 
        1. loading the data into memory
        2. loading the hyperparameters user wants to test to pick the best model
        3. set the context and target sequence processors for cleaning and tokenization
        4. set the training datasets in right format for model.fit method
        5. train the model by iterating through sets of hyperparameters loaded from step 2
        """

        # step 1
        print("loading data...")
        self.load_data(data_path)
        # step 2
        print("setting test hyperparams...")
        # self.set_test_hyperparams()
        # step 3
        print("setting processor...")
        self.set_processor()
        # step 4
        print("setting dataset...")
        self.set_train_datasets()
        # step 5
        print("training the model...")
        self.train_model()

