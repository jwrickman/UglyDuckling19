import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

class ConvAutoencoder(tf.keras.Model):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.conv_1= tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv_2= tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv_3= tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.code = tf.keras.layers.Conv2D(2, (1, 1), activation='relu', padding='same')

        self.conv_6= tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv_5= tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv_4= tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.decode = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')


        self.predict_dense_1 = tf.keras.layers.Dense(10, activation='relu')
        self.predict_dense_2 = tf.keras.layers.Dense(2, activation='softmax')

    def encode(self, inputs):
        x = self.conv_1(inputs)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = self.conv_2(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = self.conv_3(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        return self.code(x)


    def reconstruct(self, code):
        x = self.conv_4(code)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = self.conv_5(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = self.conv_6(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        return self.decode(x)

    def z_score_layer(self, x):
        x = tf.keras.layers.Flatten()(x)
        mean, variance = tf.nn.moments(x, axes=[0])
        z = (x - mean) / tf.sqrt(variance)
        return z

    def get_z_scores(self, inputs):
        code = self.encode(inputs)
        return self.z_score_layer(code)

    def predict_label(self, x):
        x = self.predict_dense_1(x)
        x = self.predict_dense_2(x)
        return x

    def call(self, inputs, training=False):
        code = self.encode(inputs)
        print(code.shape)
        reconstruction = self.reconstruct(code)
        z_scores = self.z_score_layer(code)
        predictions = self.predict_label(z_scores)
        return reconstruction, predictions
