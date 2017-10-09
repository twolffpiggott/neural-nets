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
