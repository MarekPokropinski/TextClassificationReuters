import tensorflow as tf

class LogisticRegressionModel(tf.keras.Model):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
    def call(self, x):
        return self.model(x)