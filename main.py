

import subprocess
# from facexformer.inference_cpu import test
# from facexformer.inference2 import calculate_facexformer_outputs
from facexformer.inference_cpu import test, load_model
import argparse

import glob
import shutil

import certifi
import os

from UniversalImageHandler import UniversalImageHandler

# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# os.environ['SSL_CERT_FILE'] = certifi.where()

# tasks = ["parsing", "landmarks", "headpose", "attributes", "age_gender_race", "visibility"]
tasks = ["parsing", "landmarks", "headpose"]
def main():

    # hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir="./")


    image_path = "img3.jpg"
    uih=UniversalImageHandler(image_path, debug=False)
    COMPATIBLE, img =   uih.COMPATIBLE, uih.img
    # print("COMPATIBLE:", COMPATIBLE)
    # print("shape img :", img.shape)
    # task_types= ["parsing", "landmarks", "headpose", "attributes", "age_gender_race", "visibility"]
    # calculate_facexformer_outputs(model_path, img, task_types,  device='cpu')


    image_path = "img3.jpg"
    model_path= "ckpts/model.pt"
    model=load_model(model_path)
    # results_path= "./results"
    image, faceparsing_mask, landmark_dict, headpose_dict, attributes, age_gender_race_dict, visibility_result= test(model, image_path)



if __name__ == '__main__':
    main()



