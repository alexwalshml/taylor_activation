# taylor_activation
Self-learning activation function layer compatible with tensorflow. Uses a generalized Taylor polynomial of a given order to approximate an output space.

Works as expected for simple regression models, but fails to converge in deep models. This is likely due to small gradients of the power functions, and may be alleviated with a custom training loop and a large learning rate for Taylor layers.

Note: The layers are numerically unstable with large inputs. Each TaylorActivation layer should be preceeded by a tf.keras.layers.UnitNormalization layer or any layer with a similar scaling function
