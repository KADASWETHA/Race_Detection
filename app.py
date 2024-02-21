import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
glowing_text_style = '''
    <style>
        .glowing-text {
            font-family: 'Arial Black', sans-serif;
            font-size: 33px;
            text-align: center;
            animation: glowing 2s infinite;
        }
        
        @keyframes glowing {
            0% { color: #FF9933; } /* Saffron color */
            10% { color: #FFD700; } /* Gold color */
            20% { color: #FF1493; } /* Deep Pink */
            30% { color: #00FF00; } /* Lime Green */
            40% { color: #FF4500; } /* Orange Red */
            50% { color: #9400D3; } /* Dark Violet */
            60% { color: #00BFFF; } /* Deep Sky Blue */
            70% { color: #FF69B4; } /* Hot Pink */
            80% { color: #ADFF2F; } /* Green Yellow */
            90% { color: #1E90FF; } /* Dodger Blue */
            100% { color: #FF9933; } /* Saffron color */
        }
    </style>
'''


st.markdown(glowing_text_style, unsafe_allow_html=True)
st.markdown(f'''<p class="glowing-text">
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
			cv2.putText(image,emotion_label, (x, y+100), cv2.FONT_ITALIC,1 ,(255,255,255), 1)
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
