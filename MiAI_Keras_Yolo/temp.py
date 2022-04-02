from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
from matplotlib import image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
detector = Predictor(config)
img = cv2.imread(r'T:\VNamesereceiptextract\MiAI_Keras_Yolo\filename.jpeg')
print(img)
img = Image.open(r'T:\VNamesereceiptextract\MiAI_Keras_Yolo\filename.jpeg')
box = [[135.0, 52.0], [361.0, 52.0], [361.0, 66.0], [135.0, 66.0]]
a=[box[0][0],box[0][1],box[2][0],box[2][1]]
c=img.crop(a)
text=detector.predict(c)
# x1 = top_left['x']
x1=box[0][0]
# y1 = top_left['y']
y1=box[0][1]
# x2 = bottom_right['x']
x2=box[2][0]
# y2 = top_left['y']
y2=box[0][1]
# x3 = bottom_right['x']
x3=box[2][0]
# y3 = bottom_right['y']
y3=box[2][1]
# x4 = top_left['x']
x4=box[0][0]
# y4 = bottom_right['y']
y4=box[2][1]




