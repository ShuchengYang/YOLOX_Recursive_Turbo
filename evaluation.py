import random

from tool import *

#TODO UPDATE:
version_info = '[RTND7.1]'
pre_path = 'E:/proj/'

img_path = pre_path + 'needleV1/imgs'
ann_path = pre_path + 'needleV1/annotations'

#TODO ATTENTION:
model_param_path = pre_path + 'cp/[V7.1-Recursive-Turbo]Ep38-NotBad.tar'

input_shape = 640
device = 'cuda'
num_cls = 1
nms_conf_threshold = 0.7
nms_iou_threshold = 0.5

#preperation for model param
checkpoint = torch.load(model_param_path)
#count img ttl num
imgs_folder = os.listdir(img_path)
num_imgs = len(imgs_folder)


# build up model
model = Yolox(num_cls=num_cls, training=False)
#load model param
model.load_state_dict(checkpoint['model'])
# move model to device
model.to(device)
#switch to evaluation mode
model.eval()
# l=[2775]
l = list(range(num_imgs))
# random.shuffle(l)
# random.shuffle(l)
i = 0
with torch.no_grad():
    total_exp_iou = 0.
    for i_img in l:
        cvimg_tensor = read_img(str(i_img) +'.png', img_path, device=device)
        cvpreds = model(cvimg_tensor)
        cvpreds = decode_outputs(cvpreds, [input_shape, input_shape])
        cvresults = non_max_suppression(cvpreds,
                                        num_classes=num_cls,
                                        input_shape=[input_shape, input_shape],
                                        image_shape=[input_shape, input_shape],
                                        letterbox_image=False,
                                        conf_thres=nms_conf_threshold,
                                        nms_thres=nms_iou_threshold)

        ground_truth_rec = read_ann(ann_path + '/' + str(i_img) + '.json')
        eiou = expct_iou(results=cvresults, gt_rec=ground_truth_rec)
        total_exp_iou += eiou
        i+=1
        print(f"progress {round(i/num_imgs,3)} {i_img} e {eiou} tt {total_exp_iou}")

    print(f"total exp iou {total_exp_iou}, avg exp iou {total_exp_iou/num_imgs}")
