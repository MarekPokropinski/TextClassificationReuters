import tensorflow as tf

vocab_size = 30980
num_classes = 46
maxlen = 1000

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(
    path="reuters.npz",
    num_words=vocab_size,
    skip_top=0,
    maxlen=maxlen,
    test_split=0.2,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)
ds_signature = (
    tf.TensorSpec(shape=(None,), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
)

train_ds = tf.data.Dataset.from_generator(
    lambda: ((x, y) for (x, y) in zip(x_train, y_train)),
    output_signature=ds_signature
).shuffle(x_train.shape[0])
test_ds = tf.data.Dataset.from_generator(
    lambda: ((x, y) for (x, y) in zip(x_test, y_test)),
    output_signature=ds_signature
)
