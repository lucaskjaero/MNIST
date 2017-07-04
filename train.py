import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.contrib.keras.python.keras.models import Model, Sequential
from tensorflow.contrib.keras.python.keras.layers import Activation, AveragePooling2D, BatchNormalization, concatenate, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D

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


def inception_layer(input_layer, filters):
    # Modified from Keras Documentation https://keras.io/getting-started/functional-api-guide/
    assert len(filters) == 4, "Wrong number of towers"
    tower_0_filters, tower_1_filters, tower_2_filters, tower_3_filters = filters
    assert len(tower_1_filters) == 2, "Wrong number for tower 1"
    tower_1_front, tower_1_back = tower_1_filters
    assert len(tower_2_filters) == 2, "Wrong number for tower 2"
    tower_2_front, tower_2_back = tower_2_filters

    tower_0 = Conv2D(tower_0_filters, (1, 1), padding='same', activation='relu')(input_layer)

    tower_1 = Conv2D(tower_1_front, (1, 1), padding='same', activation='relu')(input_layer)
    tower_1 = Conv2D(tower_1_back, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(tower_2_front, (1, 1), padding='same', activation='relu')(input_layer)
    tower_2 = Conv2D(tower_2_back, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    tower_3 = Conv2D(tower_3_filters, (1, 1), padding='same', activation='relu')(tower_3)

    output = concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)

    return output


def google_net():
    inputs = Input(shape=INPUT_SHAPE)

    # Input Stem
    # TODO replace BatchNormalization with LRN2D from beta release.
    convolution_layer_1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    pooling_layer_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(convolution_layer_1)
    normalization_layer_1 = BatchNormalization()(pooling_layer_1)
    convolution_layer_2 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(normalization_layer_1)
    convolution_layer_3 = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu')(convolution_layer_2)
    normalization_layer_2 = BatchNormalization()(convolution_layer_3)
    pooling_layer_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(normalization_layer_2)

    inception_layer_1 = inception_layer(pooling_layer_2, [64, [96, 128], [16, 32], 32])
    inception_layer_2 = inception_layer(inception_layer_1, [128, [128, 192], [32, 96], 64])

    pooling_layer_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception_layer_2)

    inception_layer_3 = inception_layer(pooling_layer_3, [192, [96, 208], [16, 48], 64])
    inception_layer_4 = inception_layer(inception_layer_3, [160, [112, 224], [24, 64], 64])
    inception_layer_5 = inception_layer(inception_layer_4, [128, [128, 256], [24, 64], 64])
    inception_layer_6 = inception_layer(inception_layer_5, [112, [144, 288], [32, 64], 64])
    inception_layer_7 = inception_layer(inception_layer_6, [256, [160, 320], [32, 128], 128])

    pooling_layer_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception_layer_7)

    inception_layer_8 = inception_layer(pooling_layer_4, [256, [160, 320], [32, 128], 128])
    inception_layer_9 = inception_layer(inception_layer_8, [384, [192, 384], [48, 128], 128])

    # TODO re-enable if this model doesn't perform well enough.
    #pooling_layer_5 = AveragePooling2D((7, 7), padding='valid')(inception_layer_9)

    dropout = Dropout(0.4)(inception_layer_9)

    flatten = Flatten()(dropout)
    fully_connected = Dense(10)(flatten)
    predictions = Activation('softmax')(fully_connected)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

    return model


def train_model(model):
    x_train, x_test, y_train, y_test = load_data()

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    checkpointing = ModelCheckpoint("MNIST.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', save_best_only=False, mode='auto', period=1)

    model.fit(x_train, y_train, batch_size=100, verbose=1, epochs=100, callbacks=[checkpointing, early_stopping, tensorboard], validation_data=(x_test, y_test))

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
    model = google_net()
    train_model(model)
    make_submission(model)


if __name__ == '__main__':
    main()
