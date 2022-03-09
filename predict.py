import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import random
import time

import cv2
from torchvision import transforms
from yolox import *
from utils_bbox import *
from tool import *

#TODO UPDATE:
version_info = '[RTND7.1]'
pre_path = 'E:/proj/'

infer_src_img_path = pre_path + 'needleV2/imgs'
plot_src_img_path = pre_path + 'needleV2/imgs'

res_img_saving_path = pre_path + 'cp/' + version_info + 'track'
#TODO ATTENTION:
model_param_path = pre_path + 'cp/[V7.1-Recursive-Turbo]Ep38-NotBad.tar'

input_shape = 640
device = 'cuda'
num_cls = 1
nms_conf_threshold = 0.7
nms_iou_threshold = 0.5
color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

#preperation for model param
checkpoint = torch.load(model_param_path)
#count img ttl num
imgs_folder = os.listdir(infer_src_img_path)
num_imgs = len(imgs_folder)


# build up model
model = Yolox(num_cls=num_cls, training=False)
#load model param
model.load_state_dict(checkpoint['model'])
# move model to device
model.to(device)
#switch to evaluation mode
model.eval()
with torch.no_grad():
    start = time.time()
    for i_img in range(num_imgs):
        std_pimg = read_img(str(i_img)+'.png',plot_src_img_path,tensor=False)
        cvimg_tensor = read_img(str(i_img)+'.png',infer_src_img_path,device=device)

        cvpreds = model(cvimg_tensor)
        cvpreds = decode_outputs(cvpreds, [input_shape, input_shape])
        cvresults = non_max_suppression(cvpreds,
                                        num_classes=num_cls,
                                        input_shape=[input_shape, input_shape],
                                        image_shape=[input_shape, input_shape],
                                        letterbox_image=False,
                                        conf_thres=nms_conf_threshold,
                                        nms_thres=nms_iou_threshold)
        std_pimg,_ = plot_img(std_pimg,cvresults,version_info=version_info,color=color)
        write_img(std_pimg, str(i_img)+'.png', res_img_saving_path)
        print(f"progress {round(i_img/num_imgs,3)}")

    end = time.time()
    print("predicting {} pics takes {}s, fps: {}".format(num_imgs, round(end - start, 2),
                                                         round(num_imgs / (end - start), 2)))
