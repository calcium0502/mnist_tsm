import os

from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.models import model_from_json

from my_function import *
from multiGPUCallback import MultiGPUCheckpointCallback

tfSEED = 0
npSEED = 0
PATH = os.getcwd()
gpu_count = 4
batchSize = 4096*gpu_count
tf.set_random_seed(tfSEED)  # Graphないでseed固定
epoch_num = 300

(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_mnist_data()
base_model = make_model((28, 28, 1))
base_model.summary()
# モデルの保存
json_string = base_model.to_json()
fname = PATH + "model.json"
with open(fname, "w") as f:
    f.write(json_string)


model = multi_gpu_model(base_model, gpus=gpu_count)  # マルチGPU用モデル
model.compile(optimizer=Adam(lr=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

fname = PATH + "best.hdf5"   # 最も良い物を保存
mcp = MultiGPUCheckpointCallback(filepath=fname, base_model=base_model, monitor="val_acc", verbose=2, save_best_only=True, mode="auto")

history = model.fit(train_X, train_y, batch_size=batchSize, epochs=epoch_num, verbose=2,validation_data=(val_X, val_y), callbacks=[mcp])

plot_history(history.history)
# 学習終わり
model = model_from_json(open(PATH + "model.json").read())
model.load_weights(PATH + "best.hdf5")

prediction_classification(model, test_X, test_y, plot_num=5, seed=npSEED)

