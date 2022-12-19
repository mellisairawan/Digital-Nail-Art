import streamlit as st
import os
import tempfile
import shutil
from helper import *
import PIL.Image as im

st.title('Digital Nail Art')

hand_image = im.open('hand.png')
st.image(hand_image)

hand_image = read_image('hand.png') #for open cv

color_image = st.file_uploader(
    'Upload an image of nail art color/pattern here', type=['png', 'jpeg', 'jpg'])
#color_image = read_image(color_image)

if color_image is not None:
	try:
		file_bytes = np.asarray(bytearray(color_image.read()), dtype=np.uint8)
		color_image = cv2.imdecode(file_bytes, 1)
		color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
		hand_image, color_image = resize_to_img1(hand_image, color_image)

		mask = create_nail_mask(hand_image)

		result = create_nail_color(hand_image, color_image, mask)
		st.image(result)

	except:
		st.error('Please insert nail art color/pattern image')