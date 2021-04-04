from face_detection.face_detector import face_detector
import logging
import torch
import time
import cv2
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
buffer = None


def inference(video_stream, net, confidence, cpu=False):
	# load args and create buffer folder

	# load the face mask detector model from disk
	logger.info('[INFO] loading face mask detector model...')
	detector = face_detector(net, cpu)

	# initialize the video stream and allow the camera sensor to warm up
	logger.info('[INFO] starting video stream...')
	vs = cv2.VideoCapture(video_stream)

	# loop over the frames from the video stream
	while True:
		ret, frame = vs.read()
		if not ret:
			break

		# detect face box and mask
		frame = cv2.resize(frame, (1280, 720))
		locs, preds, faces = detector.predict(frame, confidence, True)

		# loop over the detected face locations and their corresponding locations
		try:
			for i in range(len(faces)):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = locs[i]

				color = (0, 255, 0)
				label = 'Face'
				label = '{}: {:.2f}%'.format(label, preds[i])

				# display the label and bounding box rectangle on the output frame
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		except:
			pass

		# show the output frame
		cv2.imshow('Frame', frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord('q'):
			break

	# do a bit of cleanup
	vs.release()
	cv2.destroyAllWindows()