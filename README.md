# neural-nets

## Multi-label feedforward neural network with dropout and batch normalisation

```python
import feedforward
import sklearn.metrics

data = # some data, with train, validation and test split

percep = mlp.MLP(input_size=106, output_size=1, layer1_size=2048, layer2_size=1024, learn_rate=.001, activation='elu', dropout_percent=0, regu_percent=0, loss_function='binary_crossentropy')
percep.train(data[0][0].values, data[0][1].values, data[1][0].values, data[1][1].values, epochs=10, batch_size=1024)
metrics = percep.precision_recall(data[2][0].values, data[2][1].values, batch_size=1024)
fpr, tpr, thresh = sklearn.metrics.roc_curve(data[2][1].values, percep.predict(data[2][0].values, batch_size=1024))
prec, rec, thresh = sklearn.metrics.precision_recall_curve(data[2][1].values, percep.predict(data[2][0].values, batch_size=1024))

```

## Sequence to sequence LSTM
### See [fchollet's](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html "fchollet's introduction") great introduction.
```python
import helpers
from seq_to_seq import seq2seq
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '/home/tim/Github/neural-nets/fra.txt'

intexts, ttexts, inchars, tchars = helpers.get_training_data(num_samples, data_path)
encoder_input, decoder_input, decoder_target, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index, max_decoder_seq_length = helpers.preprocess_training_data(intexts, ttexts, inchars, tchars)

model = seq2seq(num_encoder_tokens, num_decoder_tokens,
                 input_token_index, target_token_index,
                 max_decoder_seq_length)
model.train(encoder_input, decoder_input, decoder_target, batch_size, epochs)

for seq_index in range(100):
    # check out decoding on a sequence
    input_seq = encoder_input[seq_index: seq_index+1]
    decoded_sentence = model.decode_sequence(input_seq)
    print('-')
    print('Input sentence: ', intexts[seq_index])
    print('Decoded sentence: ', decoded_sentence)
```
