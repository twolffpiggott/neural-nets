from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

class seq2seq:
    def __init__(self, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index, max_decoder_seq_length, latent_dim=256, optimizer='rmsprop'):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.max_decoder_seq_length = max_decoder_seq_length
        self.encoder_inputs = Input(shape=(None, num_encoder_tokens))
        self.encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = self.encoder(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]
        self.decoder_inputs = Input(shape=(None, num_decoder_tokens))
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs, initial_state=self.encoder_states)
        self.decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(decoder_outputs)
        
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)
        
        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        self.decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=self.decoder_states_inputs)
        self.decoder_states = [state_h, state_c]
        self.decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inputs] + self.decoder_states_inputs,
            [self.decoder_outputs] + self.decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

        

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size, epochs, val_split=.2):
        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2)
        self.model.save('s2s.h5')


    def decode_sequence(self, input_seq):
        # encode the input as state vectors
        states_value = self.encoder_model.predict(input_seq)
        
        # generate empty target sequence of length 1 
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # populate the first character of the target sequence with the start character
        target_seq[0, 0, self.target_token_index['\t']] = 1
        
        # sampling loop for a batch of sequences
        # assuming a batch size of 1 for simplification
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            # sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            # exit condition: either hit max length or find stop char
            if sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True
            # update the target sequence of length 1
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1
            # update the states
            states_value = [h, c]
        return decoded_sentence
