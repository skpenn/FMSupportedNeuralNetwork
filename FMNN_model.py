import tensorflow as tf
from argparse import Namespace

class Model(object):
    def __init__(self, config: Namespace):
        self.batch_size = config.batch_size
        self.field_size = config.field_size
        self.feature_size = config.feature_size
        self.factor_dim = config.factor_dim
        self.layer1_size = config.layer1_size
        self.layer2_size = config.layer2_size
        self.learning_rate = config.learning_rate

        #define input, output and parameters
        self.x = tf.placeholder("int32", shape=[self.batch_size, self.field_size])
        self.y = tf.placeholder("float", shape=[self.batch_size, 1])

        self._linear_bias = tf.Variable([tf.zeros(1)])
        self._linear_weights = tf.Variable(tf.zeros([self.feature_size, 1]))
        self._embeddings = tf.Variable(
            tf.random_normal([ self.feature_size, self.factor_dim], stddev=0.1))

        self._out_w = tf.Variable((tf.random_normal([self.layer2_size], stddev=0.1)))
        self._out_b = tf.Variable(tf.zeros(1))

        self._y = None
        self._error = None
        self._optimizer = None

    def set_pretrain_embedding(self, embedding):
        self._embeddings = tf.Variable(embedding)

    def build_model(self):
        active_vectors = tf.reshape(
            tf.nn.embedding_lookup(
                tf.concat([self._linear_weights, self._embeddings], 1),
                self.x
            ),
            [self.batch_size, (self.factor_dim+1)* self.field_size])
        active_vectors_with_bias = tf.concat(
            [
                tf.tile(self._linear_bias, [self.batch_size, 1]),
                active_vectors
            ], 1
        )

        # hidden layer 1
        layer_output = tf.nn.dropout(tf.contrib.layers.fully_connected(
            active_vectors_with_bias,
            self.layer1_size,
            activation_fn=tf.nn.tanh
        ), keep_prob=0.8)

        # hidden layer 2
        layer_output = tf.nn.dropout(tf.contrib.layers.fully_connected(
            layer_output,
            self.layer2_size,
            activation_fn=tf.nn.tanh
        ), keep_prob=0.8)

        #output = tf.reduce_sum(layer2_output, 1, keep_dims=True)
        output = tf.reduce_sum(layer_output*self._out_w+self._out_b, 1, keep_dims=True)
        self._y = tf.nn.sigmoid(output)
        self._error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=output)

        lambda_w = tf.constant(0.001, "float")
        lambda_emb = tf.constant(0.001, "float")

        # add l2 norm to avoid overfitting
        l2_norm = tf.reduce_sum(lambda_w*tf.pow(self._linear_weights, 2) + lambda_emb*tf.pow(self._embeddings, 2))
        loss = self._error + l2_norm
        self._optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def get_optimizer(self) -> tf.train.Optimizer:
        return self._optimizer

    def get_error_var(self):
        return self._error