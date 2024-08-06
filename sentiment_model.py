import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

class SentimentModel:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=True, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder_outputs = tf.keras.layers.Dense(256, activation='relu')(encoder_inputs)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(encoder_outputs)

        model = tf.keras.Model(inputs=[text_input], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, train_labels, epochs=10):
        self.model.fit(train_data, train_labels, epochs=epochs)

    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)
