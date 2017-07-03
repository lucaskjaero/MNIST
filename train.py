import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.contrib.keras.python.keras.callbacks import TensorBoard
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Conv2D, Dense, Dropout, Flatten, AveragePooling2D

HEIGHT = 28
WIDTH = 28
CHANNELS = 1
INPUT_SHAPE = (HEIGHT, WIDTH, CHANNELS)
OUTPUT_CLASSES = 10


def process_samples(samples):
    sample_number = len(samples)
    reshaped_samples = samples.as_matrix().reshape(sample_number, HEIGHT, WIDTH, CHANNELS)
    return reshaped_samples / 256


def process_labels(labels):
    return pd.get_dummies(labels).as_matrix()


def load_data():
    # Read the data
    dataset = pd.read_csv("train.csv")

    # Split into data and labels
    labels = process_labels(dataset["label"])
    samples = process_samples(dataset.drop("label", axis=1))

    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2)

    return x_train, x_test, y_train, y_test


def le_net():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), strides=1, padding='same', activation='tanh',
                     input_shape=INPUT_SHAPE))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=1, activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=1, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(OUTPUT_CLASSES, activation='relu'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def alex_net():
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 11), strides=4, padding="same", activation='relu',
                     input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"))
    model.add(Conv2D(256, kernel_size=(5, 5), strides=1, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"))
    model.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(OUTPUT_CLASSES, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model):
    x_train, x_test, y_train, y_test = load_data()
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    model.fit(x_train, y_train, batch_size=100, verbose=1, epochs=100, callbacks=[tensorboard], validation_data=(x_test, y_test))

    model.save("MNIST.h5")


def make_submission(model):
    samples = process_samples(pd.read_csv("test.csv"))
    predictions = model.predict(samples)
    category_predictions = np.argmax(predictions, axis=1)

    with open("submission.csv", 'w') as submission:
        submission.write("ImageId,Label\n")
        for index, value in enumerate(category_predictions):
            submission.write("%s,%s\n" % (index + 1, value))


def main():
    model = alex_net()
    train_model(model)
    make_submission(model)


if __name__ == '__main__':
    main()
