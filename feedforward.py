import numpy as np
import pickle
import datetime
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras import regularizers
import keras.optimizers
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import helpers

class MLP:
    def __init__(self, input_size, output_size, **kwargs):
        self.in_dim = input_size
        self.out_dim = output_size
        self.precision = None
        self.recall = None
        self.initializer = kwargs.get('initializer', 'glorot_uniform')
        self.activation = kwargs.get('activation', 'sigmoid')
        self.final_activation = kwargs.get('final_activation', 'sigmoid')
        self.dropout = kwargs.get('dropout_percent', .5)
        self.reguval = kwargs.get('regu_pecent', .0001)
        self.layer1 = kwargs.get('layer1_size', 32)
        self.layer2 = kwargs.get('layer2_size', 32)
        self.learn_rate = kwargs.get('learn_rate', .1)
        self.lr_decay_factor = kwargs.get('lr_decay_factor', .5)
        self.patience = kwargs.get('patience', 10)
        self.lossfunc = kwargs.get('loss_function', 'binary_crossentropy')

        self.model = Sequential()
        self.model.add(Dense(self.layer1, input_shape=(self.in_dim,),
                        kernel_initializer=self.initializer, kernel_regularizer=regularizers.l2(self.reguval)))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.layer2, input_shape=(self.layer1,),
                        kernel_initializer=self.initializer, kernel_regularizer=regularizers.l2(self.reguval)))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.out_dim, input_shape=(self.layer2,),
                        kernel_initializer=self.initializer, kernel_regularizer=regularizers.l2(self.reguval),
                        activation=self.final_activation))
        
        optim = keras.optimizers.Adamax(lr=self.learn_rate)
        self.model.compile(optimizer=optim,
                            loss=self.lossfunc,
                            metrics=['accuracy'])
 
    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, lr_decay=.1, log_dir=f'./tensorflow_logs/MLP_{datetime.datetime.now()}'):
        tensorboard = TensorBoard(log_dir, 
                                    histogram_freq=20,
                                    write_graph=True,
                                    write_grads=True,
                                    write_images=True)
                                    #embeddings_freq=20)
                                    
        early_stopping = EarlyStopping(monitor='loss',
                                           min_delta=0.001,
                                           patience=self.patience,
                                           verbose=1,
                                           mode='auto')
    
        plateau = ReduceLROnPlateau(monitor='val_loss',
                                        factor=lr_decay,
                                        patience=10,
                                        verbose=1)
                                        
        callbacks = [early_stopping, tensorboard, plateau]
        
        self.model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)
    
    def evaluate(self, x_test, y_test, batch_size=32):
        score = self.model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
        print(f'Test set accuracy: {score[1]}')
        return score

    def predict(self, x_test, batch_size=32):
        return self.model.predict(x_test, verbose=1, batch_size=batch_size)
        
    def precision_recall(self, x_test, y_test, nthresh=100, batch_size=32):
        pred = self.predict(x_test=x_test, batch_size=batch_size)
        thresholds = np.linspace(0, 1, num=nthresh)
        prec = np.zeros(nthresh)
        rec = np.zeros(nthresh)
        accu = np.zeros(nthresh)
        pbar = tqdm(range(len(thresholds)))
        for i, thresh in enumerate(thresholds):
            pred_bin = 1 * (pred > thresh)
            prec[i] = precision_score(np.asarray(y_test), pred_bin, average='micro')
            rec[i] = recall_score(np.asarray(y_test), pred_bin, average='micro')
            accu[i] = accuracy_score(np.asarray(y_test), pred_bin)
            pbar.update()
        pbar.close()
        self.precision = prec
        self.recall = rec
        fig, ax = plt.subplots()
        ax.plot(self.recall, self.precision, 'k--', color='blue')
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim([0, 1.1])
        ax.set_ylim([0, 1.1])
        helpers.multivac_save_graph('precision_recall_curve')
        print('thresholds, precision, recall and accuracy:')
        return np.column_stack((thresholds, prec, rec, accu))
