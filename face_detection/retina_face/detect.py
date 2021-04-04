#from __future__ import print_function
import os
import cv2
import time
import errno
import torch
import logging
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from .data import cfg_mnet, cfg_re50
from .models.retinaface import RetinaFace
from .utils.nms.py_cpu_nms import py_cpu_nms
from .layers.functions.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_keys(model, pretrained_state_dict):
    """
    Check checkpoint model Retinaface Detect
    Return True if it have checkpoint
    """
    # Check Weight
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    # Print Error
    logger.info('Missing keys:{}'.format(len(missing_keys)))
    logger.info('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    logger.info('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    """
    Load model and return model
    """
    logger.info('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if 'state_dict' in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def download_pretrain(network):
    """
        Download pretrain from AWS Server
    """
    #Create folder if it don't exist
    try:
        os.makedirs('face_detection/weights')
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    
    #Download pretrain
    if network == 'mobile0.25':
        url_mobile_1 = 'https://blueeyemodels.s3.us-east-2.amazonaws.com/Face-retina-model/mobilenet0.25_Final.pth'
        url_mobile_2 = 'https://blueeyemodels.s3.us-east-2.amazonaws.com/Face-retina-model/mobilenetV1X0.25_pretrain.tar'

        os.system('curl {} > "face_detection/weights/mobilenet0.25_Final.pth"'.format(url_mobile_1))
        os.system('curl {} > "face_detection/weights/mobilenetV1X0.25_pretrain.tar"'.format(url_mobile_2))
    
    if network == 'resnet50':
        url_resnet50 = 'https://blueeyemodels.s3.us-east-2.amazonaws.com/Face-retina-model/Resnet50_Final.pth'

        os.system('curl {} > "face_detection/weights/Resnet50_Final.pth"'.format(url_resnet50))


def get_model(network, use_cpu):
    """
    Get MobileNetV0.25 or Resnet50 Backbone
    """
    torch.set_grad_enabled(False)
    cfg = None
    flag_use_torchscript = False
    if network == 'mobile0.25':
        cfg = cfg_mnet
        trained_model = 'face_detection/weights/mobilenet0.25_Final.pth'
    elif network == 'resnet50':
        cfg = cfg_re50
        trained_model = 'face_detection/weights/Resnet50_Final.pth'
    elif network == 'traced_resnet50':
        cfg = cfg_re50
        flag_use_torchscript=True
        trained_model = torch.jit.load('traced_resnet.pt')


    # net and model
    if not flag_use_torchscript:
        try: 
            net = RetinaFace(cfg=cfg, phase = 'test', use_cpu = use_cpu)
            net = load_model(net, trained_model, use_cpu)
        except:
            download_pretrain(network)
            net = RetinaFace(cfg=cfg, phase = 'test', use_cpu = use_cpu)
            net = load_model(net, trained_model, use_cpu)
        net.eval()
    else:
        net = trained_model
        
    
    logger.info('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device('cpu' if use_cpu else 'cuda:0')
    net = net.to(device)

    return net, device, cfg


def detect_face(net, image, device, cfg):
    """
    Detect face from box return
    """
    resize = 1
    img = np.float32(image)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()
    # ignore low scores
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    # keep = nms(dets, 0.4,force_cpu=True)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:750, :]
    landms = landms[:750, :]

    dets = np.concatenate((dets, landms), axis=1)

    return dets