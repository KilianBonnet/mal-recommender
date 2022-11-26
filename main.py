import numpy as np
import tensorflow as tf

from keras import Sequential, Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense

from dataset_formatter import generate_dataset, one_hot, max_index_list


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, -1), tf.argmax(y_true, -1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


if __name__ == '__main__':
    x_train, y_train, x_test, x_train_labelled, x_test_labelled = generate_dataset()

    print("--------------------------------------------------")
    print("Train data size:", len(x_train))
    print("Test data size:", len(x_test))
    print("Data dimension:", len(x_train[0]))
    print("Example of data:", x_train[0])
    print("Score covered:", len(np.unique(y_train)), "/ 11")
    print("Total trainable features:", len(x_train) * len(x_train[0]))
    print("--------------------------------------------------")

    print("Preparing data to fit neuronal network...")
    nb_classes = 11
    y_train = one_hot(y_train, nb_classes)

    print("Building neural network...")
    model = Sequential()
    model.add(Dense(12, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    ourCallback = EarlyStopping(monitor='loss',
                                min_delta=0.0001,
                                patience=20,
                                verbose=0,
                                mode='auto',
                                baseline=None,
                                restore_best_weights=True
                                )
    model.fit(x_train, y_train, epochs=2000, batch_size=128, callbacks=[ourCallback])
    y_predicted = np.array([max_index_list(l) for l in model.predict(x_test)])

    result = []
    for i in range(0, len(y_predicted)):
        result.append((x_test_labelled[i], y_predicted[i]))

    print("Best fitting anime:")
    result = sorted(result, key=lambda row: (row[1]), reverse=True)
    for i in range(0, 10):
        print("{}, score: {}".format(result[i][0], result[i][1]))

    print()
    print("Worst fitting anime:")
    result = sorted(result, key=lambda row: (row[1]), reverse=False)
    for i in range(0, 10):
        print("{}, score: {}".format(result[i][0], result[i][1]))
