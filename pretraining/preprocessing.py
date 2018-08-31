import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import hed
curdir = os.path.abspath(os.path.curdir)
sys.path.append(os.path.join(os.path.dirname(curdir), 'tf_pose'))

from pycocotools.coco import COCO
dataset_path = '/home/artia/prj/datasets/coco'
coco = COCO(os.path.join(dataset_path, 'annotations/person_keypoints_train2017.json'))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds );

#pick a person picture ramdomly among entire training dataset.
#img_info = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

img_info = coco.loadImgs(imgIds)
filenames = [info['file_name'] for info in img_info]


images = []
for filename in filenames[:100]:
	im = cv2.imread(os.path.join(dataset_path, 'train2017', filename))
	im = cv2.resize(im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16))[None, :, :, :].astype('float32')
	print(im.shape)
	images.append(im)

images = np.array(images)

print(images.shape)
plt.imshow(images[0])

