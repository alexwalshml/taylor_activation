import tensorflow as tf


def _factorial(n):
    f = 1
    for i in range(1, n + 1):
        f *= i

    return f


class TaylorActivation(tf.keras.layers.Layer):
    def __init__(self, output_dim, order, w=None, intial='random'):
        super(TaylorActivation, self).__init__()
        self.output_dim = output_dim
        self.order = order
        self.w = w
        self.input_terms = None
        self.batch_size = None

        def taylor_initial(self):
        if self.initial == 'random':
            coefs = tf.random_normal_initializer()(shape=[self.order + 1, 1])

            return coefs

        if self.initial == 'linear':
            coefs = []
            for i in range(self.order + 1):
                if i <= 1:
                    term = 1
                else:
                    term = 0
                coefs.append(term)

            coefs = tf.constant(coefs, shape=[self.order + 1, 1], dtype=tf.float32)

            return coefs

        if self.initial == 'softplus':
            coefs = []
            with tf.GradientTape(persistent=True) as tape:
                x = tf.Variable(0.0)
                tape.watch(x)
                softplus = tf.math.log(1. + tf.math.exp(x))
                coefs.append(float(softplus))
                grad = softplus
                for i in range(1, self.order + 1):
                    grad = tape.gradient(grad, x)
                    coefs.append(float(grad) / _factorial(i))

            coefs = tf.constant(coefs, shape=[self.order + 1, 1], dtype=tf.float32)

            return coefs

        if self.initial == 'sigmoid':
            coefs = []
            with tf.GradientTape(persistent=True) as tape:
                x = tf.Variable(0.0)
                tape.watch(x)
                sigmoid = 1. / (1. + tf.math.exp(-x))
                coefs.append(float(sigmoid))
                grad = sigmoid
                for i in range(1, self.order + 1):
                    grad = tape.gradient(grad, x)
                    coefs.append(float(grad) / _factorial(i))

            coefs = tf.constant(coefs, shape=[self.order + 1, 1], dtype=tf.float32)

            return coefs
        
        if self.initial == 'tanh':
            coefs = []
            with tf.GradientTape(persistent=True) as tape:
                x = tf.Variable(0.0)
                tape.watch(x)
                tanh = tf.math.tanh(x)
                coefs.append(float(tanh))
                grad = tanh
                for i in range(1, self.order + 1):
                    grad = tape.gradient(grad, x)
                    coefs.append(float(grad) / _factorial(i))

            coefs = tf.constant(coefs, shape=[self.order + 1, 1], dtype=tf.float32)
            
            return coefs

    def build(self, input_shape):
        input_terms = 1
        for s in input_shape[1:]:
            input_terms *= s
        self.batch_size = input_shape[0]
        self.input_terms = input_terms
        if self.w is not None:
            self.w = tf.Variable(name="coefficients",
                                 initial_value=tf.random_normal_initializer()(shape=[self.order + 1, 1]),
                                 trainable=True)

    def call(self, inputs):
        taylor = None
        initial_shape = tf.shape(inputs)
        reshaped_input = tf.reshape(inputs, [initial_shape[0], self.input_terms, 1])
        for i in range(self.order + 1):
            new_term = tf.math.pow(reshaped_input, i)
            if i == 0:
                taylor = new_term
            else:
                taylor = tf.concat([taylor, new_term], axis=2)

        flat_output = tf.reduce_sum(tf.matmul(taylor, self.w), axis=2)
        output_shaped = tf.reshape(flat_output, initial_shape)

        return output_shaped
