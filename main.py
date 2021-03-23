from detect import *

source_img = 'inference_pics/GOPR0212_16050881947191.JPG'
source_img2 = 'inference_pics/IMG_20210303_104504.jpg'
weights = 'best.pt'

# conf is the confidence threshold of the detection
# iou_threshold is the area of overlap
# device, set to '' for gpu, 'cpu' for cpu
# save_txt=True saves a text file with coordinates
# save_conf=True adds the confidence level to the coordinate output
# save_img=True saves the image with boundingboxes

# for fastest result use device='', save_txt=False, save_conf=True, save_img=False
coords = detect2(source_img, weights, conf=0.7, iou_thres=0.45, device='cpu', save_txt=False, save_conf=True, save_img=False)
print(coords)

