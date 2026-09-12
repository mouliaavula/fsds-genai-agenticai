import streamlit as st 
import numpy as np 
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Elephant Image Processor", layout="wide")

# Title
st.title("Elephant Image - Multi-Color Channel Visualizer")

# Load image from URL
@st.cache_data 
def load_image():
    url = "https://media.gettyimages.com/id/103092325/photo/african-elephant-curious-about-strange-smell.jpg?s=2048x2048&w=gi&k=20&c=XU0xnaTQgp2i4yh83KWJZv1deH1HDDz3wmPlev5ktnU="
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# @st.cache_data  This is a Streamlit decorator.
#Streamlit reruns your Python script whenever the user interacts with the application.
# @st.cache_data = "If I already loaded this data, don't unnecessarily calculate/download it again."

# convert("RGB"):Open the image and make sure it is represented as RGB.
#The function finally returns this RGB image

# Load and display image
elephant = load_image()
st.image(elephant, caption="Original Elephant Image", use_container_width=True)

# Convert to NumPy array
elephant_np = np.array(elephant)
#elephant_np 3 dimensional array like (hieght,width,channel)
# Extract Red, Green and Blue channels
R, G, B = elephant_np[:, :, 0], elephant_np[:, :, 1], elephant_np[:, :, 2]

# Create channel images
# np.zeros_like() creates an array of zeros having the same shape as elephant_np.
red_img = np.zeros_like(elephant_np)
green_img = np.zeros_like(elephant_np)
blue_img = np.zeros_like(elephant_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

# Display RGB channels
st.subheader("RGB Channel Visualization")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(red_img, caption="Red Channel", use_container_width=True)

with col2:
    st.image(green_img, caption="Green Channel", use_container_width=True)

with col3:
    st.image(blue_img, caption="Blue Channel", use_container_width=True)

# Grayscale + Colormap
st.subheader("Colormapped Grayscale Image")

colormap = st.selectbox(
    "Choose a Matplotlib colormap",
    ["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "gray"]
)

elephant_gray = elephant.convert("L")
# "L" means a grayscale/luminance image in PIL.
# RGB image--->Grayscale image
# [R, G, B] i.e [255,0,0] to -->example:[120] each pixel has approximately one intensity value:

elephant_gray_np = np.array(elephant_gray)
#elephant_gray_np 2 dimensional array like (hieght,width) i.e channel ,there is no third dimension.

# RGB → 3 values per pixel
# Grayscale → 1 value per pixel

# Plot using matplotlib with colormap
fig, ax = plt.subplots(figsize=(6, 4))
# fig is entire figure/widow/canvas
# ax The actual plotting area inside the figure.

im = ax.imshow(elephant_gray_np, cmap=colormap)
# Display an array as an image.
# Color map:It tells Matplotlib:
# "How should I convert these numerical grayscale values into display colors?"

plt.axis("off")

# removes:X-axis,Y-axis,Tick marks,Axis labels

# DO NOT USE: plt.show()
# USE THIS INSTEAD:
st.pyplot(fig)

# Grayscale converts each pixel to one intensity value; a colormap only changes how those intensity values are visually represented.
