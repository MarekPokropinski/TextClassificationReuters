import tensorflow as tf
import numpy as np
from text_embeddings.bag_of_words import BagOfWords
from text_embeddings.tfidf import TFIDF, build_idf

from models.logistic_regression import LogisticRegressionModel
from data import *


bow_embeddings_train = np.zeros((x_train.shape[0], vocab_size))
bow_embeddings_test = np.empty((x_test.shape[0], vocab_size))

tfidf_embeddings_train = np.empty((x_train.shape[0], vocab_size))
tfidf_embeddings_test = np.empty((x_test.shape[0], vocab_size))

bag_of_words = BagOfWords(vocab_size)

for i in range(x_train.shape[0]):
    bow_embeddings_train[i] = bag_of_words(x_train[i])
for i in range(x_test.shape[0]):
    bow_embeddings_test[i] = bag_of_words(x_test[i])

idf = build_idf(x_train, vocab_size)

tf_idf = TFIDF(vocab_size, idf)

for i in range(x_train.shape[0]):
    tfidf_embeddings_train[i] = tf_idf(x_train[i])
for i in range(x_test.shape[0]):
    tfidf_embeddings_test[i] = tf_idf(x_test[i])
    
bow_model = LogisticRegressionModel(num_classes=46)
tfidf_model = LogisticRegressionModel(num_classes=46)
bow_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tfidf_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

bow_model.fit(bow_embeddings_train, y_train, validation_data=(bow_embeddings_test, y_test), epochs=20)
tfidf_model.fit(tfidf_embeddings_train, y_train, validation_data=(tfidf_embeddings_test, y_test), epochs=100)