import cv2
from ultralytics import YOLO
import requests
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('runs\\detect\\train\\weights')  # Make sure this points to the correct model file

# URL of the image to test
image_url = 'https://mbmart.com.vn/thumb/grande/100/329/420/products/o-to-dieu-khien-mau-vang.jpg'

# Download and read the image using requests
response = requests.get(image_url)
img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
img = cv2.imdecode(img_arr, -1)

# Run inference on the image
results = model(img)

# Loop through all the results and plot each one
for result in results:
    result.plot()  # Plot bounding boxes and labels on the image

# Display the image with the plotted results
cv2.imshow('YOLOv8 Test Image', img)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
