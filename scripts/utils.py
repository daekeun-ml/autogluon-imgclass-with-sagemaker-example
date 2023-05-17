import os
import json
import shutil
import cv2
import numpy as np
import IPython
from PIL import Image

def get_classes(raw_dir):
    classes = [d.name for d in os.scandir(raw_dir) if d.is_dir()]
    classes.sort()
    classes = set(os.listdir(raw_dir)) - set(['.ipynb_checkpoints'])
    classes_dict = {i:c for i, c in enumerate(classes)}
    return classes, classes_dict


def save_classes_dict(classes_dict, filename='classes_dict.json'):
    with open(filename, "w") as fp:
        json.dump(classes_dict, fp)


def load_classes_dict(filename):
    with open(filename, 'r') as f:
        classes_dict = json.load(f)

    classes_dict = {int(k):v for k, v in classes_dict.items()}
    return classes_dict


def get_dataset_files(raw_files, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    dataset_files = []

    for f in raw_files:
        dataset_files.append(os.path.join(os.getcwd(), dst_path, f.split('/')[-1]))
        shutil.copy2(f, dst_path)

    return dataset_files


def get_class_weights(df):
    weights = []
    num_classes = df['label'].nunique()
    for c in range(num_classes):
        class_data = df[df.label == c]
        weights.append(1 / (class_data.shape[0] / df.shape[0]))
    weights = list(np.array(weights) / np.sum(weights))
    return weights


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    f_x = x_exp / np.sum(x_exp)
    return f_x


def show_classification_result(image_path, predictor, classes_dict, scale_percent=50):
    logits = predictor.predict_proba(image_path)[0]
    probs = softmax(logits)
    pred_cls_idx = np.argmax(probs)
    pred_cls_str = classes_dict[pred_cls_idx]
    pred_cls_score = probs[pred_cls_idx]

    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(img, f'{pred_cls_str} {pred_cls_score*100:.2f}%', (10,40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    IPython.display.display(Image.fromarray(img[:,:,::-1]))
