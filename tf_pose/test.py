from pycocotools.coco import COCO
import json
with open('person_keypoints_val2017.json') as f:
	data = json.load(f)
coco = COCO('person_keypoints_val2017.json')




def search(id):
     for annotation in data['annotations']:
             if annotation['image_id']==id:
                     print(annotation)

#print(data['images'])

keys = list(coco.imgs.keys())
img_idx = coco.imgs[keys[0]]['id']

print(img_idx)
ann_idx = coco.getAnnIds(imgIds=img_idx)
print(ann_idx[0].__class__)
annotations = coco.loadAnns(ann_idx)

for ann in annotations:
	print(ann.get('num_keypoints', 0))



# image_ids = {}
# for anno in data['annotations']:
#      if anno['image_id'] in image_ids.keys():
#         image_ids[anno['image_id']] += 1
#      else:
#       	image_ids[anno['image_id']] = 0

# for i in image_ids:
# 	if  image_ids[i] > 1:
# 		print(i)

# search(8021)