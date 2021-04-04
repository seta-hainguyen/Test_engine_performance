from .retina_face.detect import detect_face, get_model
import cv2


def get_face(frame, face_confidence, dets, resize=True):
	"""
	Return: 
		faces: list of numpy array
		locs: list of tuple containing coordinates of boxes
		preds: list of scores
	"""
	faces = []
	locs = []
	preds = []
	# print(f"type:{type(dets)}, len: {len(dets)}, content:{dets}")
	# show image
	for b in dets:
		if b[4] < face_confidence:
			continue
		conf = b[4]
		b = list(map(int, b))
		startX, startY, endX, endY = b[0], b[1], b[2], b[3]
		
		try:
			face = frame[startY: endY, startX: endX]
			if resize:
				face = cv2.resize(face, (224, 224))
			preds.append(conf)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
		except:
			pass
	return faces, locs, preds


class face_detector(object):
    def __init__(self, network, cpu):
        self.net, self.device, self.cfg = get_model(network, cpu)

    def predict(self, frame, confidence, return_face=False, resize=True):
        """
        input:
                + Frame: input images type nparray
                + confidence: face's confidence
        ouput:
                + locs: location of faces
                + preds: score of each location
                + faces: faces that crop from input frame
        """
        dets = detect_face(self.net, frame, self.device, self.cfg)

        faces, locs, preds = get_face(frame, confidence, dets, resize)

        if return_face:
            return locs, preds, faces

        return locs, preds