import torch
import cv2
import numpy as np

from face_detection.retina_face.detect import get_model

net, device, cfg = get_model('resnet50', False)

cap_1 = cv2.VideoCapture('./video/camera1.avi')

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

    #tic = time.time()
    traced_net = torch.jit.trace(net, img)  # forward pass
    print(traced_net)
    return traced_net
    # logger.info('net forward time: {:.4f}'.format(time.time() - tic))

def predict(frame, confidence, return_face=False, resize=True):
    """
    input:
            + Frame: input images type nparray
            + confidence: face's confidence
    ouput:
            + locs: location of faces
            + preds: score of each location
            + faces: faces that crop from input frame
    """
    traced_net = detect_face(net, frame, device, cfg)

    return traced_net

while True:
    ret_1, frame_1 = cap_1.read()

    traced_net = predict(frame_1, 0.98, True)

    traced_net.save('traced_resnet.pt')

    print(traced_net)
    break


# test

