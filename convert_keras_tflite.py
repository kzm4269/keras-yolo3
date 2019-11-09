import tensorflow as tf
import os
import sys
import numpy as np

if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib

args = sys.argv
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args[1]  # '-1':CPU '0':GPU-0
dataset_root = argv[2]  # Cityscapesなどの画像データセット

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

converter = tf.lite.TFLiteConverter.from_keras_model_file('model_data/yolo.h5')
tflite_model = converter.convert()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# 代表的なデータセット読み込み ===================================================
# パラメータ

def representative_dataset_gen():
    from tensorflow.keras.preprocessing.image import load_img
    from pathlib import Path
    from tqdm import tqdm
    paths = list(Path(dataset_root).glob('**/*.jpg'))
    for i, path in enumerate(tqdm(paths)):
        image = np.asarray(load_img(str(path), target_size=(416, 416)))[..., :3]
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            image /= 255
        elif image.dtype == np.float64:
            image = image.astype(np.float32)
        else:
            assert image.dtype == np.float32, (path, image.dtype)
        yield [image[np.newaxis]]
    

converter.representative_dataset = representative_dataset_gen
# ================================================================================

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("./")
tflite_model_quant_file = tflite_models_dir /"integer_quant_yolo3.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)

