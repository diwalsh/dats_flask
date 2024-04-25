import cv2
import os
import numpy as np
from flask import redirect, render_template, request, session
from functools import wraps
import torch
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import imageio


def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=code, bottom=escape(message)), code


def login_required(f):
    """
    Decorate routes to require login.

    https://flask.palletsprojects.com/en/1.1.x/patterns/viewdecorators/
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

# Function to load and convert image from a uint16 to uint8 datatype.
def normalize(img_path):
    # reads the image file specified in img_path.
    # cv2.IMREAD_UNCHANGED indicates that the image is loades as-is
    # The result is stored in img and then coverted to float32.
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # pefroms mix-max normalization on the image array.
    # this normalization ensure that the pixel values are withing the range of 8-bit grayscale or RGB mages
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    # conversion to unit8 (8-bits)
    img = img.astype(np.uint8)
    # checks if the image is grayscale and, if so, converts it to an RGB image by replicating the single-channel
    # (grayscale) image to three channels (R, G, B) using np.tile()
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    return img

def resize(img):
    # here is where we can change image size prior to sending to model
    img_size = (288, 288) 
    # return resized image file 
    return cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)

# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, segmented_image):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.clip(image, 0.0, 1.0)

def model(img):
    # Define the model architecture
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")

    # Load the checkpoint weights into the model
    checkpoint_path = "static/ckpt_009.ckpt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    predictions = model(img)

    # Display the original image.
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title("Actual frame")
    plt.axis("off")

    # Display the predicted segmentation mask overlayed on the original image.
    overlayed_image = image_overlay(img, predictions)
    plt.figure(figsize=(8, 8))
    plt.imshow(overlayed_image)
    plt.title("Overlayed image")
    plt.axis("off")
    plt.show()

    # Now the model is loaded with the weights from your checkpoint
    # You can use this model instance for inference or any other purpose
    
    
    # also Inside model.py
def get_patient_images(case_prefix, day_prefix, folder_path, user_id):
    patient_images = []
    for filename in os.listdir(folder_path):
        # Check if the filename starts with the given case and day prefixes and ends with user_id
        if filename.startswith(f"case{case_prefix}_day{day_prefix}") and filename.endswith(f"_{user_id}.png"):
            patient_images.append(os.path.join(folder_path, filename))
    return patient_images


def generate_bouncing_gif(image_paths, output_gif_path):
    images = []
    for path in image_paths:
        images.append(imageio.imread(path))
    images.extend(reversed(images[1:-1]))  # Add images in reverse order, excluding the first and last
    imageio.mimsave(output_gif_path, images, duration=0.2)