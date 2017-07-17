import numpy as np
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Activation, AveragePooling2D, BatchNormalization, concatenate, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier

from .model import SqueezeNet
# SqueezeNet implementation from https://github.com/DT42/squeezenet_demo

HEIGHT = 28
WIDTH = 28
CHANNELS = 1
INPUT_SHAPE = (HEIGHT, WIDTH, CHANNELS)
OUTPUT_CLASSES = 10


def process_samples(samples, channels_first=False):
    sample_number = len(samples)
    samples_matrix = samples.as_matrix()

    if channels_first:
        reshaped_samples = samples_matrix.reshape(sample_number, CHANNELS, HEIGHT, WIDTH)
    else:
        reshaped_samples = samples_matrix.reshape(sample_number, HEIGHT, WIDTH, CHANNELS)

    return reshaped_samples / 256


def process_labels(labels):
    return pd.get_dummies(labels).as_matrix()


def load_data(channels_first=False):
    # Read the data
    dataset = pd.read_csv("train.csv")

    # Split into data and labels
    labels = process_labels(dataset["label"])
    samples = process_samples(dataset.drop("label", axis=1), channels_first=channels_first)

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


def custom_model(convolution_layers, filters, kernel_size, convolution_strides, convolution_padding, pool_size, pool_strides, pool_padding, dense_layers, dense_layer_size, dropout):
    model = Sequential()
    for layer in range(convolution_layers):
        if layer == 0:
            model.add(Conv2D(filters, kernel_size=kernel_size, strides=convolution_strides, padding=convolution_padding, activation='relu', input_shape=INPUT_SHAPE))
        else:
            model.add(Conv2D(filters, kernel_size=kernel_size, strides=convolution_strides, padding=convolution_padding, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding))
    model.add(Flatten())
    for layer in range(dense_layers):
        if layer == (dense_layers - 1):
            model.add(Dense(OUTPUT_CLASSES, activation='softmax'))
        else:
            model.add(Dense(dense_layer_size, activation='relu'))
        model.add(Dropout(dropout))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def grid_search():
    # Turns out this will take two years. Let's do this smarter.
    x_train, x_test, y_train, y_test = load_data()

    clf = KerasClassifier(custom_model, batch_size=32)

    grid = {
        'convolution_layers': [1, 2, 3, 4],
        'filters': [96, 128, 256, 384],
        'kernel_size': [(2, 2), (3, 3), (5, 5), (11, 11)],
        'convolution_strides': [1, 2, 4],
        'convolution_padding': ['same', 'valid'],
        'pool_size': [(2, 2), (3, 3), (5, 5)],
        'pool_strides': [1, 2, 4],
        'pool_padding': ['same', 'valid'],
        'dense_layers': [1, 2, 3],
        'dense_layer_size': [5, 10, 50, 100],
        'dropout': [10, 30, 50],
    }

    validator = GridSearchCV(clf, param_grid=grid, scoring='neg_log_loss', n_jobs=1)
    validator.fit(x_train, y_train)

    print('The parameters of the best model are: ')
    print(validator.best_params_)

    # validator.best_estimator_ returns sklearn-wrapped version of best model.
    # validator.best_estimator_.model returns the (unwrapped) keras model
    best_model = validator.best_estimator_.model
    metric_names = best_model.metrics_names
    metric_values = best_model.evaluate(x_test, y_test)
    for metric, value in zip(metric_names, metric_values):
        print(metric, ': ', value)

    best_model.save("BestModel.h5")

    return best_model


def train_model(model, channels_first=False):
    x_train, x_test, y_train, y_test = load_data(channels_first=channels_first)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    checkpointing = ModelCheckpoint("MNIST.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', save_best_only=False, mode='auto', period=1)

    model.fit(x_train, y_train, batch_size=100, verbose=1, epochs=100, callbacks=[checkpointing, early_stopping, tensorboard], validation_data=(x_test, y_test))

    model.save("MNIST.h5")


def make_submission(model, channels_first=False):
    samples = process_samples(pd.read_csv("test.csv"), channels_first=channels_first)
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
