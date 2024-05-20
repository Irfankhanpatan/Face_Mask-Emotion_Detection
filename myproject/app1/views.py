from django.shortcuts import render
from  django.http import HttpResponse


# Create your views here.


def Hello(request):
	return render(request,'mainpage.html') 



def run(request):
	from keras.models import load_model
	from time import sleep
	from keras.preprocessing.image import img_to_array
	from keras.preprocessing import image
	import cv2
	import numpy as np
	face_classifier = cv2.CascadeClassifier('C:/Project 4-2/myproject/app1/haarcascade_frontalface_default.xml')
	classifier =load_model('C:/Project 4-2/myproject/app1/Emotion_Detection.h5')
	class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
	#cap = cv2.VideoCapture('C:/Project 4-2/myproject/app1/video1.MP4')
	cap = cv2.VideoCapture(0)
	while True:
		try:
			ret,frame=cap.read()
			labels=[]
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			faces = face_classifier.detectMultiScale(gray,1.3,5)
			for (x,y,w,h) in faces:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
				roi_gray = gray[y:y+h,x:x+w]
				roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
				if np.sum([roi_gray])!=0:
					roi = roi_gray.astype('float')/255.0
					roi = img_to_array(roi)
					roi = np.expand_dims(roi,axis=0)
					preds = classifier.predict(roi)[0]
					label=class_labels[preds.argmax()]
					label_position = (x,y)
					cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
				else:
					cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
			cv2.imshow('Emotion Detector',frame)
			if cv2.waitKey(3)==27:
				break
		except :
			continue
	cap.release()
	cv2.destroyAllWindows()
	return render(request,'mainpage.html')
		

def mask(request):
	import cv2
	import os
	import tensorflow as tf
	from tensorflow.keras.preprocessing.image import img_to_array
	from tensorflow.keras.models import load_model
	from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
	import numpy as np
	classifier=cv2.CascadeClassifier('C:/Project 4-2/myproject/app1/harcascade.xml')
	model =tf.keras.models.load_model('C:/Project 4-2/myproject/app1/model.h5')
	video_capture = cv2.VideoCapture(0)
	#img=cv2.imread('withmask1')
	while True:
		try:
			# Capture frame-by-frame
			ret, frame = video_capture.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = classifier.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE
                                           )
			faces_list=[]
			preds=[]
			for (x, y, w, h) in faces:
				face_frame = frame[y:y+h,x:x+w]
				face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
				face_frame = cv2.resize(face_frame, (224, 224))
				face_frame = img_to_array(face_frame)
				face_frame = np.expand_dims(face_frame, axis=0)
				face_frame =  preprocess_input(face_frame)
				faces_list.append(face_frame)
				if len(faces_list)>0:
					preds = model.predict(faces_list)
					#preds=probability for mask and no mask
				for pred in preds:
					(mask, withoutMask) = pred
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
				cv2.putText(frame, label, (x, y- 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
			# Display the resulting frame
			cv2.imshow('Video', frame)
			if cv2.waitKey(3)==27:
				break
		except ValueError:
			continue
	video_capture.release()
	cv2.destroyAllWindows()
	return render(request,'mainpage.html')
	
def about(request):
	return render(request,'about.html') 

def abt(request):
	return render(request,'abt.html')


def run1(request):
	from keras.models import load_model
	from time import sleep
	from keras.preprocessing.image import img_to_array
	from keras.preprocessing import image
	import cv2
	import numpy as np
	face_classifier = cv2.CascadeClassifier('C:/Project 4-2/myproject/app1/haarcascade_frontalface_default.xml')
	classifier =load_model('C:/Project 4-2/myproject/app1/Emotion_Detection.h5')
	class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
	#cap = cv2.VideoCapture('C:/Project 4-2/myproject/app1/video1.MP4')
	if request.GET:
		var=request.GET['myFile']
		# var='C:/Project 4-2'+str(var)  
	else:
		return render(request,'abt.html')  
	cap = cv2.VideoCapture('C:/Project 4-2/'+var)
	while True:
		try:
			ret,frame=cap.read()
			labels=[]
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			faces = face_classifier.detectMultiScale(gray,1.3,5)
			for (x,y,w,h) in faces:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
				roi_gray = gray[y:y+h,x:x+w]
				roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
				if np.sum([roi_gray])!=0:
					roi = roi_gray.astype('float')/255.0
					roi = img_to_array(roi)
					roi = np.expand_dims(roi,axis=0)
					preds = classifier.predict(roi)[0]
					label=class_labels[preds.argmax()]
					label_position = (x,y)
					cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
				else:
					cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
			cv2.imshow('Emotion Detector',frame)
			while str(var[-3:]).lower()!='mp4':
				if cv2.waitKey(3)==27:
					cap.release()
					cv2.destroyAllWindows()
					return render(request,'mainpage.html')
				
			if cv2.waitKey(3)==27:
				break
		except :
			continue
	cap.release()
	cv2.destroyAllWindows()
	return render(request,'mainpage.html')




def mask1(request):
	import cv2
	import os
	import tensorflow as tf
	from tensorflow.keras.preprocessing.image import img_to_array
	from tensorflow.keras.models import load_model
	from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
	import numpy as np
	classifier=cv2.CascadeClassifier('C:/Project 4-2/myproject/app1/harcascade.xml')
	model =tf.keras.models.load_model('C:/Project 4-2/myproject/app1/model.h5')
	if request.GET:
		var=request.GET['myFile']
		# var='C:/Project 4-2'+str(var)  
	else:
		return render(request,'abt.html')  
	video_capture = cv2.VideoCapture('C:/Project 4-2/'+var)
	#img=cv2.imread('withmask1')
	while True:
		try:
			# Capture frame-by-frame
			ret, frame = video_capture.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = classifier.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE
                                           )
			faces_list=[]
			preds=[]
			for (x, y, w, h) in faces:
				face_frame = frame[y:y+h,x:x+w]
				face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
				face_frame = cv2.resize(face_frame, (224, 224))
				face_frame = img_to_array(face_frame)
				face_frame = np.expand_dims(face_frame, axis=0)
				face_frame =  preprocess_input(face_frame)
				faces_list.append(face_frame)
				if len(faces_list)>0:
					preds = model.predict(faces_list)
				for pred in preds:
					(mask, withoutMask) = pred
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
				cv2.putText(frame, label+var, (x, y- 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
			# Display the resulting frame
			cv2.imshow('Video', frame)
			while str(var[-3:]).lower()!='mp4':
				if cv2.waitKey(3)==27:
					video_capture.release()
					cv2.destroyAllWindows()
					return render(request,'mainpage.html')
			
				    

			if cv2.waitKey(3)==27:
				    break
		except ValueError:
		     	continue
	video_capture.release()
	cv2.destroyAllWindows()
	return render(request,'mainpage.html')

	 
	






















