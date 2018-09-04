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

# pick a person picture ramdomly among entire training dataset.
# img_info = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

x = coco.loadImgs(imgIds)
print('total num :', len(x))
img_info_list = x[1590:10000]
del x

# define path to HED model parameters and path to two types of output images(with/without background) 
model_path = 'HED_reproduced.npz'
output_path_with_bg = '/home/artia/prj/datasets/preprocessed_coco/edge_detected_with_bg'
output_path_without_bg = '/home/artia/prj/datasets/preprocessed_coco/edge_detected_without_bg'

if not os.path.isdir(output_path_with_bg):
	os.makedirs(output_path_with_bg)
if not os.path.isdir(output_path_without_bg):
	os.makedirs(output_path_without_bg)

i = 0
for img_info in img_info_list:
	i = i+1
	print('iteration : ', i)
	
	filename = img_info['file_name']
	im = cv2.imread(os.path.join(dataset_path, 'train2017', filename))

	base = os.path.splitext(filename)[0]
	png_filename = base + '.png'

	# create masking image with coco dataset segmentation points for filtering background. (without background)
	annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None) #get annotations ids of the image
	anns = coco.loadAnns(annIds) #get annotation infomation of that image
	contours = []
	for ann in anns:
	    if 'segmentation' in ann:
	        if type(ann['segmentation']) == list:
	            # polygon
	            for seg in ann['segmentation']:
	                poly = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
	                contours.append(poly)
	if len(contours) == 0:
		continue
	else:
		temp = im.copy()
		cv2.drawContours(temp, contours, -1, (128, 125, 10), -1)
		mask = cv2.inRange(temp, (128, 125, 10), (128, 125, 10))

		# resize images
		im = cv2.resize(im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16))[None, :, :, :].astype('float32')
		mask = cv2.resize(mask, (mask.shape[1] // 16 * 16, mask.shape[0] // 16 * 16))

		# Write a image processed with Holistic edge detection algorithm. (with background)
		edged_im = hed.run(model_path, im).astype(np.uint8)
		cv2.imwrite(os.path.join(output_path_with_bg, png_filename), edged_im)

		# # write a edged-image of which backround is white out.
		mask_inv = cv2.bitwise_not(mask)
		res = cv2.bitwise_and(edged_im, edged_im, mask=mask)
		dst = cv2.add(res, mask_inv)
		cv2.imwrite(os.path.join(output_path_without_bg, png_filename), dst)


