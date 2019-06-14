import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation, LSTM, Lambda, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.losses import binary_crossentropy as logloss

from separate import dataSeparate

def cal_accuracy(logits, targets):
    if targets.ndim != 1:
        targets = np.argmax(targets, axis=1)
    if logits.ndim != 1:
        logits = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(logits, targets))
    return 100. * num_correct / logits.shape[0]


def confusion_matrix(y_preds, y_trues):
    if len(y_trues.shape) == 2:
        y_trues = np.argmax(y_trues, axis=1)
    if len(y_preds.shape) == 2:
        y_preds = np.argmax(y_preds, axis=1)

    size = y_trues.max()+1
    confusion_mat = np.zeros([size, size])

    for y_pred, y_true in zip(y_preds, y_trues):
        confusion_mat[y_true, y_pred] += 1

    return confusion_mat


def prediction_classification(model, test_X, test_y, plot_num=0, seed=None):
    np.random.seed(seed)
    """
    model_fname = dir_name+"teach_model.json"
    model = model_from_json(open(model_fname).read())
    weight_fname = dir_name+"teach_best.hdf5"
    model.load_weights(weight_fname)
    """
    predict_y = model.predict(test_X, verbose=0)

    accuracy = cal_accuracy(predict_y, test_y)
    print("accuracy :", accuracy)

    con_mat = confusion_matrix(predict_y, test_y)
    plt.figure()
    plt.imshow(con_mat, cmap="jet", interpolation="nearest", vmin=0)
    plt.colorbar()

    for i in range(plot_num):
        rnd = np.random.randint(test_X.shape[0])
        y_data = np.argmax(test_y[rnd, :])
        prediction = np.argmax(predict_y[rnd, :])
        Str = "label :{0} || prediction :{1}".format(y_data, prediction)
        plt.figure()
        plt.title(label=Str)
        # plt.plot(x_data[0, :], "k-", label="x")
        # plt.plot(x_data[1, :], "r-", label="y")
        # plt.plot(x_data[2, :], "b-", label="z")
        plt.plot(test_X[rnd, :, 0], "k-", label="x")
        plt.plot(test_X[rnd, :, 1], "r-", label="y")
        plt.plot(test_X[rnd, :, 2], "b-", label="z")
        plt.xlabel("Time")
        plt.ylabel("Accelarate")
        plt.legend()
        plt.grid(which="major", color="black", linestyle="-")
        plt.grid(which="minor", color="black", linestyle="--")
    plt.show()


def plot_history(history, acc=True, val=True):
    num = len(history["loss"])

    plt.figure()
    plt.plot(range(1, num+1), history["loss"], "k-", label="training")
    if val:
        plt.plot(range(1, num+1), history["val_loss"], "r--", label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(which="major", color="black", linestyle="-")
    plt.grid(which="minor", color="black", linestyle="--")
    plt.yscale('log')

    if acc:
        plt.figure()
        plt.plot(range(1, num+1), history["acc"], "k-", label="training")
        if val:
            plt.plot(range(1, num+1), history["val_acc"], "r--", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(which="major", color="black", linestyle="-")
        plt.grid(which="minor", color="black", linestyle="--")


def make_model(input_shape):
    tfSEED = 0
    # ResNet

    def Plain(inputs_x, size=8):
        s = size // 2
        x = Conv1D(size, 5, padding="SAME")(inputs_x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)

        x = Conv1D(size, 5, padding="SAME")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)

        x = add([inputs_x, x])
        return Activation("relu")(x)

    def Bottleneck(inputs_x, size=16):
        s = size // 4
        x = Conv2D(s, 1, padding="SAME")(inputs_x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)

        x = Conv2D(s, 5, padding="SAME")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)

        x = Conv2D(size, 1, padding="SAME")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)

        x = add([inputs_x, x])
        return Activation("relu")(x)

    # 教師モデル
    with tf.device("/cpu:0"):
        # モデルの構築は明示的にCPUで
        tf.set_random_seed(tfSEED)

        # Channel last
        # 1データごとのサイズ
        inputs = Input(shape=input_shape, name="main_inputs")

        x = Conv2D(8, 1, padding="SAME")(inputs)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        # x = Dropout(0.5)(x)
        # x = LSTM_block(x)
        for _ in range(4):
            x = Bottleneck(x, size=8)
        x = Conv2D(16, 8, strides=8, padding="SAME")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        for _ in range(8):
            x = Bottleneck(x, size=16)

        x = Conv2D(32, 5, strides=5, padding="SAME")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        for _ in range(4):
            x = Bottleneck(x, size=32)

        flat1 = Flatten()(x)
        fc = Dense(32)(flat1)
        bn2 = BatchNormalization()(fc)
        relu = Activation("relu", name="hidden")(bn2)
        fc2 = Dense(10)(relu)
        outputs = Activation("softmax")(fc2)  # 後で分離するので層を分ける？
        return Model(inputs=inputs, outputs=outputs)


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    ds = dataSeparate([0.5, 0.5], x_test.shape[0])
    x_val, x_test = ds.separate(x_test)
    y_val, y_test = ds.separate(y_test)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_val = x_val.reshape(-1, 28, 28, 1).astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")

    x_train /= 255
    x_val /= 255
    x_test /= 255

    class_num = 10

    y_train = keras.utils.to_categorical(y_train, class_num)
    y_val = keras.utils.to_categorical(y_val, class_num)
    y_test = keras.utils.to_categorical(y_test, class_num)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
