import io
import os
import base64
import torch
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from PIL import Image
from dataclasses import dataclass
from helpers import normalize
from flask import Flask, redirect, render_template, request, session
from flask_session import Session




@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES:   int = 4 # including background.
    IMAGE_SIZE: tuple[int,int] = (288, 288) # W, H
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD:  tuple = (0.229, 0.224, 0.225)
    BACKGROUND_CLS_ID: int = 0

@dataclass
class InferenceConfig:
    BATCH_SIZE:  int = 10
    NUM_BATCHES: int = 2

# Create a mapping of class ID to RGB value.
id2color = {
    0: (0, 0, 0),    # background pixel
    1: (0, 0, 255),  # Stomach
    2: (0, 255, 0),  # Small Bowel
    3: (255, 0, 0),  # large Bowel
}

def num_to_rgb(num_arr, color_map=id2color):
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))
 
    for k in color_map.keys():
        output[single_layer == k] = color_map[k]
 
    # return a floating point array in range [0.0, 1.0]
    return np.float32(output) / 255.0

def resize_segmented_image(original_image, segmented_image):
    # Resize segmented image to match the dimensions of the original image
    resized_segmented_image = cv2.resize(segmented_image, (original_image.shape[1], original_image.shape[0]))
    return resized_segmented_image

# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, segmented_image):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.
    
    # both to BGR
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert images to float32 for compatibility
    image = image.astype(np.float32)
    segmented_image = segmented_image.astype(np.float32)
    
    # overlay, and then back to RGB
    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.clip(image, 0.0, 1.0)

def overlay_images(original_image, mask_image, alpha=1.0, beta=0.7, gamma=0.0):
    # Convert images to BGR format (OpenCV uses BGR by default)
    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)

    # Overlay mask on top of the original image
    blended_image = cv2.addWeighted(original_bgr, alpha, mask_bgr, beta, gamma)

    # Convert the blended image back to RGB
    blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)

    return blended_image_rgb



@torch.inference_mode()
def inference(model, image_paths, device="cpu"):
    batch_size = InferenceConfig.BATCH_SIZE

    num_images = len(image_paths)
    num_batches_to_process = (num_images + batch_size - 1) // batch_size

    results = []
    overlay_paths = []

    for batch_idx in range(num_batches_to_process):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_paths))

        batch_images = []

        for path in image_paths[start_idx:end_idx]:
            # Load image from path
            image = Image.open(path).convert("RGB")
            # Convert image to numpy array
            image_np = np.array(image)
            # Convert image to tensor -- and make sure a float and not double!!
            image_tensor = torch.tensor(image_np / 255.0, dtype=torch.float).permute(2, 0, 1).to(device)
       
            batch_images.append(image_tensor)
        batch_images = torch.stack(batch_images)

        model.eval()
        # Perform inference
        with torch.no_grad():
            predictions = model(batch_images)

        pred_all = predictions.logits.argmax(dim=1)

        batch_results = []
        mask_paths = []

        for i in range(len(batch_images)):
            # Save original image
            image_folder = 'static/originals'
            os.makedirs(image_folder, exist_ok=True)
            image_name = os.path.basename(image_paths[start_idx + i])[:-4] + ".png"  # Remove extension
            image_path = os.path.join(image_folder, image_name)
            Image.open(image_paths[start_idx + i]).save(image_path, format='PNG')
            
            # Save segmented image
            mask_folder = 'static/masks'
            os.makedirs(mask_folder, exist_ok=True)
            mask_name = os.path.basename(image_paths[start_idx + i])[:-4] + "_mask.png"  # Remove extension and add "_mask.png"
            mask_path = os.path.join(mask_folder, mask_name)
            mask_paths.append(mask_path)
            Image.fromarray((num_to_rgb(pred_all[i], color_map=id2color) * 255).astype(np.uint8)).save(mask_path, format='PNG')

            # for debugging purposes to print shapes
            original_image = np.array(Image.open(image_path))
            segmented_mask = np.array(Image.open(mask_path))
            # print("Original image shape:", original_image.shape)
            # print("Segmented mask shape:", segmented_mask.shape)

            # resizing mask shape to fit original images
            resized_mask = resize_segmented_image(original_image, num_to_rgb(pred_all[i], color_map=id2color))
            print("Resized mask shape:", resized_mask.shape)
            
            # Save resized segmented image
            resized_mask_folder = 'static/resized'
            os.makedirs(resized_mask_folder, exist_ok=True)
            mask_name = os.path.basename(image_paths[start_idx + i])[:-4] + "_mask.png"  # Remove extension and add "_mask.png"
            mask_path = os.path.join(resized_mask_folder, mask_name)
            # mask_paths.append(mask_path)
            Image.fromarray((resized_mask * 255).astype(np.uint8)).save(mask_path, format='PNG')

            # TRYING SOMETHING NEW!!!!!!!
            original_image = np.array(Image.open(image_path))
            resized_mask = np.array(Image.open(mask_path))
            overlayed_image = overlay_images(original_image, resized_mask)
             
            # Overlay predicted mask on the original image -- doesn't work at all :(
            # overlayed_image = image_overlay(original_image, resized_mask)
            
            print("Overlaid image shape:", overlayed_image.shape)

            # Save the overlayed image for debugging
            overlay_folder = 'static/overlaid'
            overlay_name = os.path.basename(image_paths[start_idx + i])[:-4] + "_mask.png"  # Remove extension and add "_mask.png"
            overlay_path = os.path.join(overlay_folder, overlay_name)
            # overlayed_image_path = 'static/overlaid/overlayed_image_{}.png'.format(start_idx + i)
            Image.fromarray((overlayed_image * 255).astype(np.uint8)).save(overlay_path, format='PNG')
            print("Overlayed image saved:", overlay_path)
            overlay_paths.append(overlay_path)
            

            # Convert overlayed image to bytes
            image_bytes = io.BytesIO()
            Image.fromarray((overlayed_image * 255).astype(np.uint8)).save(image_bytes, format='PNG')
            image_bytes.seek(0)
            
            # Encode image bytes as base64 string
            base64_image = base64.b64encode(image_bytes.read()).decode('utf-8')

            # Get formatted name
            parts = image_paths[start_idx + i].split("/")[-1].split("_")
            case_number = ''.join(filter(str.isdigit, parts[0]))
            day_number = ''.join(filter(str.isdigit, parts[1]))
            slice_number = int(parts[3])
            formatted_name = "Case {}; Day {}: Slice {}".format(case_number, day_number, slice_number)
                
            batch_results.append((base64_image, formatted_name, image_paths[start_idx + i]))

        results.extend(batch_results)

    overlay_paths.sort()
    print(overlay_paths)
    # Store image paths in session
    session['image_paths'] = overlay_paths
        
    # Sort results by path
    results.sort(key=lambda x: x[2])
    return results

# @torch.inference_mode()
# def inference(model, image_paths, device="cpu"):
#     batch_size = InferenceConfig.BATCH_SIZE

#     num_images = len(image_paths)
#     num_batches_to_process = (num_images + batch_size - 1) // batch_size

#     results = []

#     for batch_idx in range(num_batches_to_process):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(image_paths))

#         batch_results = []

#         for i in range(start_idx, end_idx):
#             path = image_paths[i]
#             # Load image from path
#             image = Image.open(path).convert("RGB")
#             # Convert image to numpy array
#             image_np = np.array(image)
#             # Convert image to tensor -- and make sure a float and not double!!
#             image_tensor = torch.tensor(image_np / 255.0, dtype=torch.float).permute(2, 0, 1).to(device)
       
#             # Perform inference
#             model.eval()
#             with torch.no_grad():
#                 predictions = model(image_tensor.unsqueeze(0))

#             pred_all = predictions.logits.argmax(dim=1)

#             # Save segmented image to static/masks folder
#             mask_folder = 'static/masks'
#             os.makedirs(mask_folder, exist_ok=True)
#             mask_name = os.path.basename(path)[:-4] + "_mask.png"  # Remove extension and add "_mask.png"
#             mask_path = os.path.join(mask_folder, mask_name)
#             Image.fromarray((num_to_rgb(pred_all[0], color_map=id2color) * 255).astype(np.uint8)).save(mask_path, format='PNG')

#             # Convert segmented mask to Base64 encoded image data
#             mask_image = Image.open(mask_path)
#             buffered = io.BytesIO()
#             mask_image.save(buffered, format="PNG")
#             mask_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

#             # Get formatted name
#             parts = os.path.basename(path).split("_")
#             case_number = ''.join(filter(str.isdigit, parts[0]))
#             day_number = ''.join(filter(str.isdigit, parts[1]))
#             slice_number = int(parts[3])
#             formatted_name = "Case {}; Day {}: Slice {}".format(case_number, day_number, slice_number)

#             batch_results.append((mask_image_base64, formatted_name, path))

#         results.extend(batch_results)
#     # Sort results by path
#     results.sort(key=lambda x: x[2])
#     return results


def load_model(checkpoint_path):
    # Load the configuration from the pre-trained model
    config = SegformerConfig.from_pretrained('nvidia/segformer-b4-finetuned-ade-512-512', num_labels=4)

    # Initialize the model with the modified configuration
    model = SegformerForSemanticSegmentation(config)

    # Modify the last layers to match the number of classes your task has
    model.decode_head.classifier = torch.nn.Conv2d(768, 4, kernel_size=(1, 1))
    model.decode_head.classifier.bias = torch.nn.Parameter(torch.zeros(4))
    
    # Assuming you are using PyTorch Lightning to manage the training
    checkpoint = torch.load(checkpoint_path, map_location='mps')
    
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict({k.replace('model.', ''): v for k, v in model_state_dict.items()}, strict=False)

    return model

