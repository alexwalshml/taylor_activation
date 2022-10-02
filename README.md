# taylor_activation
Self-learning activation function layer compatible with tensorflow. Uses a generalized Taylor polynomial of a given order to approximate an output space. Efficacy results pending.

# Requirements
`tensorflow==2.9.1`

`sympy==1.11.1`

While it isn't strictly required, you will also need

`tensorflow_addons==0.17.1`

# Usage
A `TaylorActivation` layer can be used in place of any activation function in a tensorflow model. The polynomial layer should be compiled using the custom `TaylorAdam` optimizer included with the use of `tensorflow_addons.optimizers.Multioptimizer`. 

## Parameters
`order`: Required. An integer corresponding to the degree of the polynomial approximation. Larger arguments may cause performance issues and can more easily lead to overfitting. Recommended to keep `order <= 10`

`initial`: Optional. A string or `sympy.Expr` to use to calculate the initial polynomial coefficients. Currently accepted strings are 'random', 'linear', 'softplus', 'sigmoid', and 'tanh'. Default is 'random', but all other provided values yield better performance.

`clip_in`: Optional. A float used to clip input tensor values by their absolute value. Usage not recommended.

`clip_out`: Optional. A float used to clip output tensor values by their absolute value. Usage not recommended.

## Example
Below is an example of a basic classifier. A more in depth example can be found [here](example/script/link), and a full tutorial notebook with functioning tests can be found [here](tutorial/notebook/link).

```
from tensorflow import keras
from keras import layers, Sequential, optimizers
from tensorflow_addons.optimizers import MultiOptimizer

model = Sequential([layers.Input(size=(5,)),
                    layers.Dense(128),
                    layers.ReLU(),
                    layers.Dense(128),
                    layers.ReLU(),
                    layers.Dense(1),
                    TaylorActivation(order=9, initial='sigmoid')]
                    
optimizer = MultiOptimizer([(optimizers.Adam(), model.layers[:-1]),
                              (TaylorAdam(), [model.layers[-1]])]

model.compile(loss='bce', optimizer=optimizer)
```

# TaylorAdam
Included in the module is a custom version of the Adam optimizer to ensure proper fitting of high-order polynomials. Arguments are identical to [tf.keras.optimizers.Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) except for `amsgrad`, which is enabled in `TaylorAdam`.
