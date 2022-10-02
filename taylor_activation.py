import tensorflow as tf
import numpy as np
from sympy import log, tanh, Expr, Symbol, exp, Poly
from tensorflow import keras
from tensorflow.python.ops import math_ops, array_ops
from keras import layers
from typing import Union


# static function to ensure tailing zero coefficients are included
def _order_check(c, n):
    if len(c) != n + 1:
        c = np.append(c, 0.)

    return c.reshape((-1, 1))


# computes the factorial
def _factorial(n):
    f = 1
    for i in range(1, n + 1):
        f *= i

    return f


# clips a tensor by its absolute value
def _clip_by_abs(x, clip):
    x = tf.clip_by_value(x, -clip, clip)

    return x


class TaylorActivation(layers.Layer):
    """
    Usage: Add a TaylorActivation layer to any keras model as you would any other layer.
           TaylorActivation layers act as self-learning activation functions with a set
           of nonlinear weights. Caution must be used with the input and output spaces of
           this layer to ensure there are no overflows. The clip_in and clip_out parameters
           have been included to prevent this, but their use should be minimized as they
           will reduce variance of propagated values.
    Args:
        order: An integer to be the degree of the polynomial to be used in the calculations
        initial: A predefined string or sympy.Expr to be used in calculating the
                 initial values of the series coefficients. Allowed string values
                 are currently 'random', 'linear', 'sigmoid', 'softplus', and 'tanh'
        symbol: A sympy.Symbol to differentiate against in the case of a user-defined
                initial function. Defaults to None
        clip_in: A float representing the maximum absolute value allowed for
                 input values. Larger absolute values will be clipped to this.
                 Defaults to np.inf
        clip_out: A float representing the maximum absolute value allowed for
                  output values. Larger absolute values will be clipped to this.
                  Defaults to np.inf
    """

    def __init__(self,
                 order: int,
                 initial: Union[str, Expr] = 'random',
                 symbol: Symbol = None,
                 clip_in: float = np.inf,
                 clip_out: float = np.inf):
        super(TaylorActivation, self).__init__()
        self.order = order
        self.initial = initial
        self.symbol = symbol
        self.clip_in = clip_in
        self.clip_out = clip_out
        self.w = None  # parameter to store trainable weights once initialized

    def taylor_initializer(self):
        if self.initial == 'random':
            # initializes w randomly
            # not recommended
            coefs = tf.random_normal_initializer()(shape=[self.order + 1, 1])

            return coefs

        elif self.initial == 'linear':
            # initializes a linear term, but sets all other coefficients to zero
            # this represents a linearly scaled output
            coefs = []
            for i in range(self.order + 1):
                if i == 1:
                    term = 1
                else:
                    term = 0
                coefs.append(term)

            coefs = tf.constant(coefs, shape=[self.order + 1, 1], dtype=tf.float32)

            return coefs

        else:
            # if initial is a string, the corresponding sympy function is utilized
            # if the string is not in the approved list, a ValueError is returned
            x = Symbol('x')

            if self.initial == 'softplus':
                func = log(1. + exp(x))
            elif self.initial == 'sigmoid':
                func = 1. / (1. + exp(-x))
            elif self.initial == 'tanh':
                func = tanh(x)
            elif isinstance(self.initial, str):
                raise ValueError
            else:
                # if initial is a sympy function, it is assigned to func
                # and converted to a series
                func = self.initial
                x = self.symbol

            coefs = Poly(func.series(x, 0, self.order + 1).removeO()).all_coeffs()

            coefs = _order_check(np.flip(coefs), self.order)  # sympy floats are incompatible with tensorflow

            # coefficients are cast to a tensor and returned
            coefs = tf.constant(coefs, shape=[self.order + 1, 1], dtype=tf.float32)

            return coefs

    def build(self, input_shape):
        # overrides the build() method required in layer definitions
        # assigns trainable parameters
        self.w = tf.Variable(name="coefficients",
                             initial_value=self.taylor_initializer(),
                             trainable=True)

    def call(self, inputs):
        # overrides the call() method required in layer definitions
        # performs calculations and returns layer output
        taylor = None
        initial_shape = tf.shape(inputs)

        # inputs are clipped if a value was specified
        reshaped_input = _clip_by_abs(tf.reshape(inputs, [initial_shape[0], -1, 1]), self.clip_in)

        # for each term in the polynomial, the inputs are raised to the corresponding power
        for i in range(self.order + 1):
            new_term = tf.ones(shape=tf.shape(reshaped_input))
            # we use repeated instances of tf.multiply instead of a single instance of tf.pow
            # tf.pow introduces numerical instabilities in the gradient
            # whereas tf.multiply does not
            for j in range(i):
                new_term = tf.multiply(new_term, reshaped_input)
            if i == 0:
                taylor = new_term
            else:
                taylor = tf.concat([taylor, new_term], axis=2)

        flat_output = tf.reduce_sum(tf.matmul(taylor, self.w), axis=2)

        # outputs are clipped to avoid overflow
        flat_output = _clip_by_abs(flat_output, self.clip_out)
        output_shaped = tf.reshape(flat_output, initial_shape)

        return output_shaped

    def get_config(self):
        # overrides get_config() method
        config = super(TaylorActivation, self).get_config()
        config.update({
            'order': self.order,
            'initial': self.initial,
            'symbol': self.symbol,
            'safety': self.safety,
        })


class TaylorAdam(keras.optimizers.Optimizer):
    """
    Usage: An optimizer to apply scaled learning rates to learned polynomial
           weights. The amsgrad variant of the Adam algorithm is used here.

           IMPORTANT: Do not use TaylorAdam with any non-TaylorActivation
           layers. The scaled learning rate will cause most of the parameters
           to recieve little to no change. One should utilize
           tfa.optimizers.MultiOptimizer to ensure that each layer is
           properly trained.
    Args:
        learning_rate: A float to be used as the initial learning rate of the
        algorithm.
        beta_1: A float to be used as the exponential decay rate for the
        first moment estimates.
        beta_2: A float to be used as the exponential decay rate for the
        second moment estimates.
        epsilon: A small float to be used to avoid division by zero
    """

    def __init__(self,
                 learning_rate: float = 0.0001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-7,
                 name: str = "TaylorAdam",
                 **kwargs):
        super().__init__(name, **kwargs)
        # parameters are assigned as hyperparameters
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        # For each variable, a slot is created for it
        # three consecutive loops are used to preserve ordering
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        for var in var_list:
            self.add_slot(var, "v_hat")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        # updates the slots and performs a single optimization step

        # learning rate scaling tensor is declared with similarity to taylor series formula
        fact_arr = tf.constant([1. / _factorial(n) for n in range(var.shape[0])], shape=[var.shape[0], 1])
        var_dtype = var.dtype.base_dtype

        # variables are updated according to amsgrad variant of Adam
        lr_d = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1 = self._get_hyper("beta_1", var_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        v_hat = self.get_slot(var, "v_hat")

        m_t = m * beta_1 + (1. - beta_1) * grad
        v_t = v * beta_2 + (1. - beta_2) * grad ** 2
        v_hat_t = math_ops.maximum(v_t, v_hat)

        lr_t = lr_d * tf.sqrt(1. - beta_2_power) / (1. - beta_1_power)

        var_t = var - lr_t * m_t * fact_arr / (tf.sqrt(v_hat_t) + self.epsilon)

        # updated variables are assigned to their slots
        m.assign(m_t)
        v.assign(v_t)
        v_hat.assign(v_hat_t)
        var.assign(var_t)

    def get_config(self):
        # overrides the get_config() optimizer method
        config = super(TaylorAdam, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._initial_decay,
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
        })

        return config
