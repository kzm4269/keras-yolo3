import tensorflow as tf
import os
import sys
import numpy as np
#from FileRW import CFileRW

if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib

args = sys.argv
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args[1]  # '-1':CPU '0':GPU-0

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

converter = tf.lite.TFLiteConverter.from_keras_model_file('model_data/yolo.h5')
tflite_model = converter.convert()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 代表的なデータセット読み込み ===================================================
# パラメータ


def representative_dataset_gen():
    WIDTH = 512
    HEIGHT = 512
    DATANUM = 200
    SAMPLE = 10

    # read data
    FileRW = CFileRW()
    # input data
    input = FileRW.read(filename="../../../data/mult_new/10/Input_x512y512z1_hp16.raw", datatype="float32")
    print("input:", input.shape, input.dtype)

    # reshape
    # input data
    #input = input.reshape(DATANUM, HEIGHT, WIDTH, 1)
    input = input.reshape(1, HEIGHT, WIDTH, 1)
    print("input:", input.shape, input.dtype)

    input_data = np.zeros(shape=(1,HEIGHT,WIDTH, 1), dtype="float32")
    input_data = np.array(input, dtype="float32")

    yield [input_data]

    # for i in range(int(DATANUM/SAMPLE)):
    #     choice = np.arange(DATANUM)
    #     print(choice)
    #     choice = np.random.choice(a=choice, size=choice.shape[0], replace=False)
    #     print(choice)
    #     input = input[choice]
    #     print(i)
    #     yield [input]

converter.representative_dataset = representative_dataset_gen
# ================================================================================

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("./")
tflite_model_quant_file = tflite_models_dir /"integer_quant_yolo3.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)

