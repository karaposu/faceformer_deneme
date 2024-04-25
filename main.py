

import subprocess
from facexformer.inference_cpu import test
import argparse
import glob
import shutil

import certifi
import os

os.environ['SSL_CERT_FILE'] = certifi.where()


tasks = ["parsing", "landmarks", "headpose", "attributes", "age_gender_race", "visibility"]
def main():
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    image_path= "img3.jpg"
    # results_path="./"
    results_path="/Users/ns/Desktop/projects/faceformer_deneme/"
    for task in tasks:
        flag = 1
        args = {

            "model_path": "/Users/ns/.cache/torch/hub/checkpoints/swin_b-68c6b09e.pth",
            "image_path": image_path,
            "results_path": results_path,
            "task": task,
            "gpu_num": "0"
        }
        args_namespace = argparse.Namespace(**args)
        test(args_namespace)


if __name__ == '__main__':
    main()



