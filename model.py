import tensorflow as tf
import sys

EPS = 1e-12


# Class for batch normalization node
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            reuse=tf.AUTO_REUSE  # if tensorflow vesrion < 1.4, delete this line
                                            )


# leaky relu function
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


def feature_loss(real_feats, fake_feats):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = tf.square(tf.reduce_mean(real_feat) - tf.reduce_mean(fake_feat))
        losses += l2
    return losses


class Generator:
    def __init__(self):
        # Parameters
        # Encoding
        self.ch_G0 = 3
        self.ch_G1 = 64
        self.ch_G2 = 128
        self.ch_G3 = 256
        self.ch_G4 = 512
        # Decoding
        self.ch_G5 = 256
        self.ch_G6 = 128
        self.ch_G7 = 64
        self.ch_G8 = 3

        # Chennels
        # Encoding
        self.G_W1 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G0, self.ch_G1], stddev=0.02), name="G_W1")

        self.G_W2 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G1, self.ch_G2], stddev=0.02), name='G_W2')
        self.G_bn2 = batch_norm(name="G_bn2")

        self.G_W3 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G2, self.ch_G3], stddev=0.02), name='G_W3')
        self.G_bn3 = batch_norm(name="G_bn3")

        self.G_W4 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G3, self.ch_G4], stddev=0.02), name='G_W4')
        self.G_bn4 = batch_norm(name="G_bn4")

        # Decoding
        self.G_W5 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G5, self.ch_G4], stddev=0.02), name='G_W5')
        self.G_bn5 = batch_norm(name="G_bn5")

        self.G_W6 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G6, self.ch_G5], stddev=0.02), name='G_W6')
        self.G_bn6 = batch_norm(name="G_bn6")

        self.G_W7 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G7, self.ch_G6], stddev=0.02), name='G_W7')
        self.G_bn7 = batch_norm(name="G_bn7")

        self.G_W8 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G8, self.ch_G7], stddev=0.02), name='G_W8')

        # param set
        self.params = [
            self.G_W1,
            self.G_W2,
            self.G_W3,
            self.G_W4,
            self.G_W5,
            self.G_W6,
            self.G_W7,
            self.G_W8
        ]

    def generate(self, input_img, batch_size):
        h1 = tf.nn.conv2d(input_img, self.G_W1, strides=[1, 2, 2, 1], padding='SAME')  # [?,64,64,3] -> [?,32,32,64]
        h1 = lrelu(h1)

        h2 = tf.nn.conv2d(h1, self.G_W2, strides=[1, 2, 2, 1], padding='SAME')  # [?,32,32,64] -> [?,16,16,128]
        h2 = self.G_bn2(h2)
        h2 = lrelu(h2)

        h3 = tf.nn.conv2d(h2, self.G_W3, strides=[1, 2, 2, 1], padding='SAME')  # [?,16,16,128] -> [?,8,8,256]
        h3 = self.G_bn3(h3)
        h3 = lrelu(h3)

        h4 = tf.nn.conv2d(h3, self.G_W4, strides=[1, 2, 2, 1], padding='SAME')  # [?,8,8,256] -> [?,4,4,512]
        h4 = self.G_bn4(h4)
        h4 = lrelu(h4)

        h5 = tf.nn.conv2d_transpose(h4, self.G_W5, output_shape=[batch_size, 8, 8, self.ch_G5], strides=[1, 2, 2, 1])  # [?,4,4,512] -> [?,8,8,256]
        h5 = self.G_bn5(h5)
        h5 = tf.nn.relu(h5)

        h6 = tf.nn.conv2d_transpose(h5, self.G_W6, output_shape=[batch_size, 16, 16, self.ch_G6], strides=[1, 2, 2, 1])  # [?,8,8,256] -> [?,16,16,128]
        h6 = self.G_bn6(h6)
        h6 = tf.nn.relu(h6)

        h7 = tf.nn.conv2d_transpose(h6, self.G_W7, output_shape=[batch_size, 32, 32, self.ch_G7], strides=[1, 2, 2, 1])  # [?,16,16,128] -> [?,32,32,64]
        h7 = self.G_bn7(h7)
        h7 = tf.nn.relu(h7)

        h8 = tf.nn.conv2d_transpose(h7, self.G_W8, output_shape=[batch_size, 64, 64, self.ch_G8], strides=[1, 2, 2, 1])  # [?,32,32,64] -> [?,64,64,3]
        h8 = tf.nn.tanh(h8)

        return h8

    def sample_generator(self, input_image, imshape, sess, batch_size=1):
        input_img = tf.placeholder(tf.float32, [batch_size] + imshape)

        h1 = tf.nn.conv2d(input_img, self.G_W1, strides=[1, 2, 2, 1], padding='SAME')  # [?,64,64,3] -> [?,32,32,64]
        h1 = lrelu(h1)

        h2 = tf.nn.conv2d(h1, self.G_W2, strides=[1, 2, 2, 1], padding='SAME')  # [?,32,32,64] -> [?,16,16,128]
        h2 = self.G_bn2(h2)
        h2 = lrelu(h2)

        h3 = tf.nn.conv2d(h2, self.G_W3, strides=[1, 2, 2, 1], padding='SAME')  # [?,16,16,128] -> [?,8,8,256]
        h3 = self.G_bn3(h3)
        h3 = lrelu(h3)

        h4 = tf.nn.conv2d(h3, self.G_W4, strides=[1, 2, 2, 1], padding='SAME')  # [?,8,8,256] -> [?,4,4,512]
        h4 = self.G_bn4(h4)
        h4 = lrelu(h4)

        h5 = tf.nn.conv2d_transpose(h4, self.G_W5, output_shape=[batch_size, 8, 8, self.ch_G5], strides=[1, 2, 2, 1])  # [?,4,4,512] -> [?,8,8,256]
        h5 = self.G_bn5(h5)
        h5 = tf.nn.relu(h5)

        h6 = tf.nn.conv2d_transpose(h5, self.G_W6, output_shape=[batch_size, 16, 16, self.ch_G6], strides=[1, 2, 2, 1])  # [?,8,8,256] -> [?,16,16,128]
        h6 = self.G_bn6(h6)
        h6 = tf.nn.relu(h6)

        h7 = tf.nn.conv2d_transpose(h6, self.G_W7, output_shape=[batch_size, 32, 32, self.ch_G7], strides=[1, 2, 2, 1])  # [?,16,16,128] -> [?,32,32,64]
        h7 = self.G_bn7(h7)
        h7 = tf.nn.relu(h7)

        h8 = tf.nn.conv2d_transpose(h7, self.G_W8, output_shape=[batch_size, 64, 64, self.ch_G8], strides=[1, 2, 2, 1])  # [?,32,32,64] -> [?,64,64,3]
        h8 = tf.nn.tanh(h8)

        generated_samples = sess.run(h8, feed_dict={input_img: input_image})
        return generated_samples


class Discriminator:
    def __init__(self):
        # Channels
        self.ch_D0 = 3
        self.ch_D1 = 64
        self.ch_D2 = 128
        self.ch_D3 = 256
        self.ch_D4 = 512
        self.ch_D5 = 1

        # Parameters
        self.D_W1 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D0, self.ch_D1], stddev=0.02), name='D_W1')

        self.D_W2 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D1, self.ch_D2], stddev=0.02), name='D_W2')
        self.D_bn2 = batch_norm(name="D_bn2")

        self.D_W3 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D2, self.ch_D3], stddev=0.02), name='D_W3')
        self.D_bn3 = batch_norm(name="D_bn3")

        self.D_W4 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D3, self.ch_D4], stddev=0.02), name='D_W4')
        self.D_bn4 = batch_norm(name="D_bn4")

        self.D_W5 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D4, self.ch_D5], stddev=0.02), name='D_W5')

        self.params = [
            self.D_W1,
            self.D_W2,
            self.D_W3,
            self.D_W4,
            self.D_W5
        ]

    def discriminate(self, input_img):
        h1 = tf.nn.conv2d(input_img, self.D_W1, strides=[1, 2, 2, 1], padding='SAME')  # [?,64,64,3] -> [?,32,32,64]
        h1 = lrelu(h1)

        h2 = tf.nn.conv2d(h1, self.D_W2, strides=[1, 2, 2, 1], padding='SAME')  # [?,32,32,64] -> [?,16,16,128]
        h2 = self.D_bn2(h2)
        h2 = lrelu(h2)

        h3 = tf.nn.conv2d(h2, self.D_W3, strides=[1, 2, 2, 1], padding='SAME')  # [?,16,16,128] -> [?,8,8,256]
        h3 = self.D_bn3(h3)
        h3 = lrelu(h3)

        h4 = tf.nn.conv2d(h3, self.D_W4, strides=[1, 2, 2, 1], padding='SAME')  # [?,8,8,256] -> [?,4,4,512]
        h4 = self.D_bn4(h4)
        h4 = lrelu(h4)

        h5 = tf.nn.conv2d(h4, self.D_W5, strides=[1, 1, 1, 1], padding='VALID')  # [?,4,4,512] -> [?,1,1,1]
        h5 = tf.nn.sigmoid(h5)

        Feature = [h2, h3, h4]

        return h5, Feature


class Discogan:
    # Network Parameters
    def __init__(self, sess, batch_size):
        self.learning_rate = 0.0002
        self.weight_decay = 0.00001
        self.decay_gan_loss = 10000
        self.starting_rate = 0.01
        self.changed_rate = 0.5

        self.sess = sess
        self.batch_size = batch_size
        self.image_shape = [64, 64, 3]

        self._build_model()

    # Build the Network
    def _build_model(self):
        self.x_A = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        self.x_B = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        self.rate = tf.placeholder(tf.float32)

        self.G_AB = Generator()
        self.G_BA = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()

        x_AB = self.G_AB.generate(self.x_A, self.batch_size)
        x_ABA = self.G_BA.generate(x_AB, self.batch_size)
        discrim_AB, Feature_AB = self.D_B.discriminate(x_AB)
        discrim_B, Feature_B = self.D_B.discriminate(self.x_B)
        L_FEATURE_B = feature_loss(Feature_B, Feature_AB)
        L_CONST_A = tf.reduce_mean(tf.losses.mean_squared_error(x_ABA, self.x_A))
        L_GAN_B = tf.reduce_mean(-tf.log(discrim_AB + EPS))
        L_G_AB = (L_GAN_B * 0.1 + L_FEATURE_B * 0.9) * (1.0 - self.rate) + L_CONST_A * self.rate
        L_D_B = tf.reduce_mean(-tf.log(discrim_B + EPS) - tf.log(1 - discrim_AB + EPS))

        x_BA = self.G_BA.generate(self.x_B, self.batch_size)
        x_BAB = self.G_AB.generate(x_BA, self.batch_size)
        discrim_BA, Feature_BA = self.D_A.discriminate(x_BA)
        discrim_A, Feature_A = self.D_A.discriminate(self.x_A)
        L_FEATURE_A = feature_loss(Feature_A, Feature_BA)
        L_CONST_B = tf.reduce_mean(tf.losses.mean_squared_error(x_BAB, self.x_B))
        L_GAN_A = tf.reduce_mean(-tf.log(discrim_BA + EPS))
        L_G_BA = (L_GAN_A * 0.1 + L_FEATURE_A * 0.9) * (1.0 - self.rate) + L_CONST_B * self.rate
        L_D_A = tf.reduce_mean(-tf.log(discrim_A + EPS) - tf.log(1 - discrim_BA + EPS))

        self.L_G = L_G_AB + L_G_BA
        self.L_D = L_D_A + L_D_B

        self.gen_params = self.G_AB.params + self.G_BA.params
        self.discrim_params = self.D_A.params + self.D_B.params

        regularizer_G = tf.zeros(tf.shape(self.L_G))
        for param in self.gen_params:
            regularizer_G += tf.nn.l2_loss(param)

        regularizer_D = tf.zeros(tf.shape(self.L_D))
        for param in self.discrim_params:
            regularizer_D += tf.nn.l2_loss(param)

        # regularization
        self.L_G += self.weight_decay * regularizer_G
        self.L_D += self.weight_decay * regularizer_D

        self.train_op_discrim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.L_D, var_list=self.discrim_params)
        self.train_op_gen = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.L_G, var_list=self.gen_params)

    def sample_generate(self, input_image, direction, batch_size=1):
        if direction == 'AB':
            return self.G_AB.sample_generator(input_image, self.image_shape, self.sess, batch_size)
        elif direction == 'BA':
            return self.G_BA.sample_generator(input_image, self.image_shape, self.sess, batch_size)
        elif direction == 'ABA':
            return self.G_BA.sample_generator(self.G_AB.sample_generator(input_image, self.image_shape, self.sess, batch_size), self.image_shape, self.sess, batch_size)
        elif direction == 'BAB':
            return self.G_AB.sample_generator(self.G_BA.sample_generator(input_image, self.image_shape, self.sess, batch_size), self.image_shape, self.sess, batch_size)
        else:
            sys.exit("direction should be 'AB' or 'BA'")

    # Train Generator and return the loss
    def train_gen(self, xA, xB, iterr):
        if iterr < self.decay_gan_loss:
            rate = self.starting_rate
        else:
            rate = self.changed_rate

        _, loss_G = self.sess.run([self.train_op_gen, self.L_G], feed_dict={self.x_A: xA, self.x_B: xB, self.rate: rate})
        return loss_G

    # Train Discriminator and return the loss
    def train_discrim(self, xA, xB, iterr):
        if iterr < self.decay_gan_loss:
            rate = self.starting_rate
        else:
            rate = self.changed_rate

        _, loss_D = self.sess.run([self.train_op_discrim, self.L_D], feed_dict={self.x_A: xA, self.x_B: xB, self.rate: rate})
        return loss_D
