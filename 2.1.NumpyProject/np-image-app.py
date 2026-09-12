import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  
# PIL stands for Python Imaging Library.
# Image provides functions to open, display,resize,crop,covert etc   
# image = Image.open("photo.jpg") opens an image from a file
import requests
# requests is a Python library used to communicate with websites/URLs.
from io import BytesIO

# Helper function to load image from a URL
def load_image_from_url(url):
    response = requests.get(url) # Go to this URL and download whatever the server sends back."
    return Image.open(BytesIO(response.content))

# image = Image.open("photo.jpg") opens an image from a file but if the image is downloaded from the internet.
#The downloaded image is stored in memory as bytes. BytesIO allows us to treat those bytes like a file.

# response.content: This gives us the raw bytes of the downloaded image.
# BytesIO takes those bytes and makes them behave like a file in memory.

# Elephant image URL
elephant_url = "https://media.gettyimages.com/id/103092325/photo/african-elephant-curious-about-strange-smell.jpg?s=2048x2048&w=gi&k=20&c=XU0xnaTQgp2i4yh83KWJZv1deH1HDDz3wmPlev5ktnU="
#elephant_url= 'https://plus.unsplash.com/premium_photo-1700752733721-bceab91f2b86?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8cGVhY29jayUyMGZlYXRoZXJ8ZW58MHx8MHx8fDA%3D'

# Load elephant image
elephant = load_image_from_url(elephant_url)

# Display original image
plt.figure(figsize=(6, 4))
plt.imshow(elephant)
plt.title("Elephant")
plt.axis("off")
plt.show()

# Convert to NumPy array and print shape
elephant_np = np.array(elephant)
print("Elephant image shape:", elephant_np.shape)

# Convert to grayscale
elephant_gray = elephant.convert("L")

# Display grayscale image
plt.figure(figsize=(6, 4))
plt.imshow(elephant_gray, cmap="gray")
plt.title("Elephant (Grayscale)")
plt.axis("off")
plt.show()

