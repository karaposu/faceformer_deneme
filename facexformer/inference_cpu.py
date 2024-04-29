
import os
import sys

script_dir = os.path.dirname(__file__)  # Directory of the current script
sys.path.append(script_dir)  # Appe
sys.path.append(os.path.join(os.path.dirname(__file__), 'facexformer'))
print("script_dir: ", script_dir)



import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
import argparse
from math import cos, sin
from PIL import Image
from network import FaceXFormer
from facenet_pytorch import MTCNN
from facexformer_utils import denorm_points, unnormalize, adjust_bbox, visualize_head_pose, visualize_landmarks, visualize_mask
from task_postprocesser import task_faceparsing,     process_landmarks, task_headpose , task_attributes, task_gender, process_visibility

import os



def load_model(weights_path):
    gpu_num = 0
    device = torch.device("cuda:" + str(gpu_num) if torch.cuda.is_available() else "cpu")

    model = FaceXFormer().to(device)
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict_backbone'])
        print("Model loaded successfully.")
    except Exception as e:
        print("Failed to load model:", e)
        return None
    model.eval()
    return model


def test(model, image_path):

    device = torch.device("cuda:" + str(gpu_num) if torch.cuda.is_available() else "cpu")

    transforms_image = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224,224), interpolation=InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    mtcnn = MTCNN(keep_all=True)
    image = Image.open(image_path)
    width, height = image.size
    boxes, probs = mtcnn.detect(image)
    x_min, y_min, x_max, y_max = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
    x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, width, height)

    image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    print("image size after crop", image.size)
    # (346, 438)

    image = transforms_image(image)

    label_shapes = {
        "segmentation": (224, 224),
        "lnm_seg": (5, 2),
        "landmark": (68, 2),
        "headpose": (3,),
        "attribute": (40,),
        "a_g_e": (3,),
        "visibility": (29,)
    }
    labels = {key: torch.zeros(shape) for key, shape in label_shapes.items()}
    data = {
        "image": image,
        "label": labels,
    }

    # images, labels, tasks = data["image"], data["label"], data["task"]
    images, labels = data["image"], data["label"]
    for k in labels.keys():
        labels[k] = labels[k].unsqueeze(0).to(device=device)
    images = images.unsqueeze(0).to(device=device)


    for i in range(6):
         task = torch.tensor([i])
         task = task.to(device=device)
         output = model(images, labels, task)

         for e in output:
            print(e.shape)
         landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output =output
         if task[0] == 0:
             faceparsing_mask=task_faceparsing(seg_output)
         if task[0] == 1:
             landmark_dict=process_landmarks(landmark_output,images )
         if task[0] == 2:
             headpose_dict=task_headpose(headpose_output, images)
         if task[0] == 3:
             attributes=task_attributes(attribute_output)
         if task[0] == 4:
             age_gender_race_dict=task_gender(age_output,gender_output,race_output )
         if task[0] == 5:
             visibility_result=process_visibility(visibility_output)


    image = unnormalize(images[0].detach().cpu())
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    image =image[:, :, ::-1]

    # save_path = os.path.join(results_path, "face.png")
    # cv2.imwrite(f"{save_path}", image[:, :, ::-1])
    return image, faceparsing_mask, landmark_dict, headpose_dict, attributes, age_gender_race_dict, visibility_result

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, help="Provide absolute path to your weights file")
    # parser.add_argument("--image_path", type=str, help="Provide absolute path to the image you want to perform inference on")
    # parser.add_argument("--results_path", type=str, help="Provide path to the folder where results need to be saved")
    # parser.add_argument("--task", type=str, help="parsing" or "landmarks" or "headpose" or "attributes" or "age_gender_race" or "visibility")
    # parser.add_argument("--gpu_num", type=str, help="Provide the gpu number")
    # args = parser.parse_args()
    # test(args, model_path,image_path, results_path  )
    pass