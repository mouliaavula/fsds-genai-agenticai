import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.set_page_config(page_title='Image processor from local file system', layout='wide')
st.title("Multi Color Channel Visualizer")

@st.cache_data
def load_image():
    path = r"E:\vs-code-prakash\data\hourse.avif"
    return Image.open(path).convert('RGB')

hourse_image = load_image()
st.image(hourse_image, caption='Original Image',use_container_width=True)

hourse_np = np.array(hourse_image)
print(f'shape of original image {hourse_np.shape}')
print(f'dimension of orginal image {hourse_np.ndim}')

R,G,B = hourse_np[:,:,0],hourse_np[:,:,1],hourse_np[:,:,2]

print(f'shape of each channel {R.shape}')
print(f'dimension of each channel {R.ndim}')

red_image = np.zeros_like(hourse_np)
green_image = np.zeros_like(hourse_np)
blue_image = np.zeros_like(hourse_np)

red_image[:,:,0] = R
green_image[:,:,1] = G
blue_image[:,:,2] = B

print(f'shape of each channel after update {red_image.shape}')
print(f'dimension each channel after update {red_image.ndim}')

st.subheader('RGB Channel visualization')
col1, col2, col3 = st.columns(3)

with col1:
    st.image(red_image,caption='Red Channel',use_container_width=True)
with col2:
    st.image(green_image,caption='Green Channel',use_container_width=True)
with col3:
    st.image(blue_image, caption='Blue Channel',use_container_width=True)


st.subheader('Color mapped gray scale image')
color_map = st.selectbox(
    "Choose a Matplotlib colormap",
    ["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "gray"]
)

hourse_gray_image = hourse_image.convert('L')
hourse_gray_np = np.array(hourse_gray_image)

fig, ax = plt.subplots(figsize=(6,4))
ax.imshow(hourse_gray_np,cmap=color_map)
ax.axis("off")
st.pyplot(fig)