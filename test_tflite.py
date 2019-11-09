import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import matplotlib.pyplot as plt

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image


def predict_keras(model_path):
    model = keras.models.load_model(model_path, compile=False)
    
    def predict(image):
        assert image.ndim == 3, image.shape
        assert image.dtype == np.float32, image.dtype
        assert image.ptp() <= 1.0, image.ptp()
        return model.predict([image[None]])
    
    return predict


def predict_tflite(model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    def predict(image):
        assert image.ndim == 3, image.shape
        assert image.dtype == np.float32, image.dtype
        assert image.ptp() <= 1.0, image.ptp()
        
        # Test model on random input data.
        print('- predict_tflite: interpreter.set_tensor')
        interpreter.set_tensor(input_details[0]['index'], image[None])
        
        print('- predict_tflite: interpreter.invoke')
        interpreter.invoke()
        
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        print('- predict_tflite: interpreter.get_tensor')
        return [interpreter.get_tensor(output_ditail['index']) for output_ditail in output_details]
    
    return predict
        
        
def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='model path (.h5 or .tflite)')
    parser.add_argument('images', nargs='+', help='image paths')
    args = parser.parse_args()
    
    anchors = np.reshape(list(map(int, Path('./model_data/yolo_anchors.txt').read_text().strip().split(','))), (-1, 2))
    class_names = Path('./model_data/coco_classes.txt').read_text().strip().splitlines()

    predict = {
        'h5': predict_keras,
        'tflite': predict_tflite,
    }[args.model.split('.')[-1]](args.model)
    
    for i, image_path in enumerate(map(Path, args.images)):
        print('load:', image_path)
        pil_image = Image.open(str(image_path))
        input_data = letterbox_image(pil_image, size=(416, 416))
        input_data = input_data / np.float32(255.)
        image = np.asarray(pil_image)
        # image = input_data.copy()
        
        print('predict:', image_path)
        output_data = predict(input_data)
        
        print('eval:', image_path)
        result = yolo_eval(
            [keras.backend.constant(d) for d in output_data],
            anchors=anchors, 
            num_classes=len(class_names),
            image_shape=(image.shape[0], image.shape[1]),
            score_threshold=0.3,
            iou_threshold=0.45,
        )
        boxes, scores, classes = [keras.backend.eval(t) for t in result]
        print('boxes =', boxes)
        
        print('save:', image_path)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = FigureCanvasAgg(plt.Figure()).figure
        ax = fig.add_subplot(1,1,1)
        ax.imshow(image)
        for i, (top, left, bottom, right) in enumerate(boxes):
            assert top <= bottom and left <= right
            ax.add_patch(plt.Rectangle(xy=[left, top], width=right - left, height=bottom - top, fill=False, linewidth=3, color='red'))
        fig.savefig(f'out_{args.model.split(".")[-1]}_{i:03d}.png')
        
        
if __name__ == '__main__':
    _main()