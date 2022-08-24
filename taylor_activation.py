import tensorflow as tf


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
            coefs = []
            for i in range(self.order + 1):
                i_factorial = 1.
                for m in range(1, i + 1):
                    i_factorial *= m
                term = np.random.normal() / i_factorial
                coefs.append(term)

            coefs = tf.constant(coefs, shape=[self.order + 1, 1], dtype=tf.float32)

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
            domain = np.linspace(-1., 1., 20001)
            softplus = np.log(1. + np.exp(domain))
            h = np.diff(domain)[0]
            coefs = []
            for i in range(self.order + 1):
                i_factorial = 1.
                for m in range(1, i + 1):
                    i_factorial *= m
                zero_point = 0
                for j in range(i):
                    zero_point += ((-1.) ** (i + j)) \
                                  * (np.math.factorial(i) / (np.math.factorial(j) * np.math.factorial(i - j))) \
                                  * softplus[10000 + j]
                    zero_point /= h ** i
                term = zero_point / i_factorial
                coefs.append(term)

            coefs = tf.constant(coefs, shape=[self.order + 1, 1], dtype=tf.float32)

            return coefs

        if self.initial == 'sigmoid':
            domain = np.linspace(-1., 1., 20001)
            sigmoid = 1. / (1. + np.exp(-domain))
            h = np.diff(domain)[0]
            coefs = []
            for i in range(self.order + 1):
                i_factorial = 1.
                for m in range(1, i + 1):
                    i_factorial *= m
                zero_point = 0
                for j in range(i):
                    zero_point += ((-1.) ** (i + j)) \
                                  * (np.math.factorial(i) / (np.math.factorial(j) * np.math.factorial(i - j))) \
                                  * sigmoid[10000 + j]
                    zero_point /= h ** i
                term = zero_point / i_factorial
                coefs.append(term)

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
