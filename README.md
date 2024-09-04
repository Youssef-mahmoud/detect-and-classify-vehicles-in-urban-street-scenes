# Vehicle Detection and Classification System
![Alt text](https://drive.google.com/uc?export=view&id=184iKWGvrIiXhROLJhlerDcAwlcfjsOrs)
![Alt text](https://drive.google.com/uc?export=view&id=1reis1SerSD9YYp_9q2tSBa5VKAXsUWjE)


Welcome to the demo notebook for the Computer Vision Engineer Assessment Task. In this notebook, we present a comprehensive solution for detecting and classifying vehicles in urban street scenes. Our goal is to develop a robust system that accurately identifies and categorizes vehicles using advanced computer vision techniques. Below is a summary of our approach:

## Dataset
We utilize the COCO dataset, concentrating on the 'car', 'truck', and 'bus' categories. This diverse dataset offers a wide range of urban street scenes, providing a solid foundation for training and evaluating our vehicle detection system.

## Object Detection and Vehicle Classification
We employ YOLO (You Only Look Once), a state-of-the-art object detection algorithm, known for its real-time performance and high accuracy in object localization. YOLO allows us to detect vehicles within images efficiently. Once detected, the vehicles are classified into three categories: cars, trucks, and buses. This integrated approach ensures precise identification and categorization of vehicles.

## Deployment
To demonstrate our system, we have created a user-friendly interface that allows users to upload images, run predictions, and view the results interactively.

We aim for this notebook to showcase the effectiveness and practicality of our vehicle detection and classification system, highlighting its capabilities and potential applications.

# Dependencies & Installations

```bash
!pip install ultralytics
!pip install ipywidgets
```

```bash
import ipywidgets as widgets
from IPython.display import display, HTML
from PIL import Image
import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
import os
```
# Loading the model
You can download the YOLOv8 model from [here](https://drive.google.com/file/d/1mYZdaD02OHnE0kf3bHjpn_lZxjdixUum/view?usp=drive_link).

```bash
model = YOLO('/content/yolov8n.pt')
```
# Deployment
```bash
# Create widgets for image upload and display
upload_widget = widgets.FileUpload(
    accept='image/*',  # Accept only image files
    multiple=False     # Allow only a single file upload at a time
)

# Create widgets to display images
output_image = widgets.Image()
uploaded_image_output = widgets.Output()
prediction_image_output = widgets.Output()
```

```bash
def show_image(image, output_widget, title=''):
    """
    Display an image with a title in a specified output widget.

    Args:
    - image: The image to display (PIL Image or numpy array).
    - output_widget: The widget to display the image in.
    - title: The title for the image (string).
    """
    with output_widget:
        output_widget.clear_output(wait=True)  # Clear previous output
        plt.figure(figsize=(10, 10))  # Set the figure size
        plt.imshow(image)             # Display the image
        plt.title(title)              # Set the title
        plt.axis('off')               # Hide the axis
        plt.show()
```

```bash
def handle_upload(change):
    """
    Handle the image upload, perform prediction, and display results.

    Args:
    - change: The change event triggered by the file upload widget.
    """
    if upload_widget.value:
        # Read the uploaded image
        uploaded_image = list(upload_widget.value.values())[0]['content']
        image = Image.open(io.BytesIO(uploaded_image))
        image_np = np.array(image)

        # Display the uploaded image
        show_image(image, uploaded_image_output, title='Uploaded Image')

        # Convert PIL image to OpenCV format (BGR color space)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform prediction using the model
        results = model.predict(source=image_cv, save=True, classes=[2, 5, 7], conf=0.5)

        # Locate the saved result images
        output_dir = 'runs/detect/predict'
        result_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg') or f.endswith('.png')]

        if result_files:
            # Load the first result image
            result_image_path = os.path.join(output_dir, result_files[0])
            result_image = Image.open(result_image_path)
            show_image(result_image, prediction_image_output, title='Prediction Result')
        else:
            with prediction_image_output:
                prediction_image_output.clear_output(wait=True)
                print("No result images found.")

```

```bash
# Attach the handler function to the upload widget
upload_widget.observe(handle_upload, names='value')

# Display the interface elements
display(HTML("<h2>Upload an Image for Vehicle Detection and Classification</h2>"))
display(upload_widget)
display(uploaded_image_output)
display(prediction_image_output)
```
