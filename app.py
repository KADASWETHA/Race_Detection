import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
st.markdown(f'''
<h1 align='center'>Race Detection System</h1>''',unsafe_allow_html=True)


cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

user_image=st.file_uploader('Please upload your image',type=['jpg','jpeg','png'])
btn=st.button('Predict')
if btn and user_image is not None:
	bytes_data = user_image.getvalue()
	image= cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
	result=DeepFace.analyze(image,actions=['race'])
	gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = cascade.detectMultiScale(gray_frame, 1.1, 3)
	for x,y,w,h in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (4, 29, 255), 2, cv2.LINE_4)
		user_selected_items = list(result[0].keys())
		if 'dominant_race' in user_selected_items:
			emotion_label='Race: '+str(result[0]['dominant_race']).title()
			st.write(emotion_label)
			cv2.putText(image,'white', (x, y+h+110), cv2.FONT_ITALIC,1 ,(255,255,255), 2)
	col1,col2=st.columns(2)
	with col1:
		st.info('Original Image')
		st.image(user_image,use_column_width=True)
	with col2:
		st.info('Detected Image')
		st.image(image, use_column_width=True,channels='BGR')
		
		st.markdown(f'''<h4 align='center'>Detected Race: {result[0]['dominant_race']}</h1>''',

		unsafe_allow_html=True)
elif btn and user_image is None:
	st.warning('Please Check the file')