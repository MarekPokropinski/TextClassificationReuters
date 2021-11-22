import tensorflow as tf


class LstmModel(tf.keras.Model):
    def __init__(self, num_classes, vocab_size, embeddings) -> None:
        super().__init__()
        self.num_classes = num_classes
        if embeddings is not None:
            embedding_layer = tf.keras.layers.Embedding(
                vocab_size, 50,
                tf.keras.initializers.Constant(embeddings),
                embeddings_regularizer=tf.keras.regularizers.L1(l1=0.001)
            )
            # embedding_layer.trainable = False
        else:
            embedding_layer = tf.keras.layers.Embedding(vocab_size, 50, embeddings_regularizer=tf.keras.regularizers.L1(l1=0.001))
        self.model = tf.keras.models.Sequential([
            embedding_layer,
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

    def call(self, x):
        return self.model(x)
