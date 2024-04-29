import numpy as np
import cv2
import base64
import urllib3
import re
from PIL import Image, UnidentifiedImageError
import io

class UniversalImageHandler:
    def __init__(self, img_input, img_is_a_mask=False, debug=False):
        self.img_input = img_input
        self.img_is_a_mask = img_is_a_mask
        self.img = None
        self.COMPATIBLE = False
        self.debug=debug

        if self.debug:
            print("debug On")

        self.read_image()

    def adjust_image_channels(self, img):
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # Remove the alpha channel if present
        if self.img_is_a_mask and img.ndim == 3:
            img = img[:, :, 0]  # Use the first channel for masks
        return img

    def read_image(self):

        if self.debug:
            print("checking image input type")
            print(self.img_input)

        if isinstance(self.img_input, np.ndarray):

            if self.debug :
                print("input is ndarray")
            self.process_image(self.img_input)
        elif isinstance(self.img_input, str):
            print("input is string")
            if self.is_url(self.img_input):
                if self.debug:
                    print("input is url")
                self.handle_url_image(self.img_input)
            elif self.is_path(self.img_input):
                if self.debug:
                    print("input is path")
                self.handle_path_image(self.img_input)
            elif self.is_base64(self.img_input):
                if self.debug:
                    print("input is base64 image")
                self.handle_base64_image(self.img_input)

    def handle_url_image(self, url):
        try:
            user_agent = {'user-agent': 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) ..'}
            http = urllib3.PoolManager(10, headers=user_agent)
            response = http.urlopen('GET', url)
            image = Image.open(io.BytesIO(response.data))
            img_arr = np.array(image)
            self.process_image(img_arr)
        except (UnidentifiedImageError, urllib3.exceptions.HTTPError) as e:
            print(f"Failed to load image from URL: {e}")

    def handle_path_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.process_image(img)
        else:
            print("Failed to load image from path.")

    def handle_base64_image(self, encoded_img):
        try:
            decoded_img = base64.b64decode(encoded_img)
            img_np_arr = np.frombuffer(decoded_img, np.uint8)
            img = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)
            self.process_image(img)
        except ValueError:
            print("Invalid Base64 encoding.")

    def process_image(self, img):
        img = self.adjust_image_channels(img)
        self.img = img
        self.COMPATIBLE = True

    # def is_path(self, s):
    #     path_regex = re.compile(
    #         r'^(/|\\|[a-zA-Z]:\\|\.\\|..\\|./|../)'
    #         r'(?:(?:[^\\/:*?"<>|\r\n]+\\|[^\\/:*?"<>|\r\n]+/)*'
    #         r'[^\\/:*?"<>|\r\n]*)$',
    #         re.IGNORECASE)
    #     return re.match(path_regex, s) is not None

    def is_path(self, s):
        path_regex = re.compile(
            r'^(/|\\|[a-zA-Z]:\\|\.\\|..\\|./|../)?'  # Optional start with /, \, C:\, .\, ..\, ./, or ../
            r'(?:(?:[^\\/:*?"<>|\r\n]+\\|[^\\/:*?"<>|\r\n]+/)*'  # Directory names
            r'[^\\/:*?"<>|\r\n]*)$',  # Last part of the path which can be a file
            re.IGNORECASE)
        return re.match(path_regex, s) is not None

    def is_url(self, s):
        url_regex = re.compile(
            r'^(https?://|ftp://)'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(url_regex, s) is not None

    def is_base64(self, s):
        try:
            s = s.strip()
            if len(s) % 4 != 0:
                return False
            base64.b64decode(s, validate=True)
            return True
        except ValueError:
            return False


















# import urllib3
# import logging
# from PIL import Image,UnidentifiedImageError
# import io
# import re
# import numpy as np
# import base64
# import cv2
#
#
#
#
#
# class ImageObject:
#     def __init__(self, img_input, img_is_a_mask=False):
#         self.IMG_IS_NP_ARRAY = False
#         self.IMG_HAS_3_CHANNEL = False
#         self.IMG_HAS_4_CHANNEL = False
#         self.IMG_IS_BASE64_ENCODED = False
#         self.IMG_IS_LINK = False
#         self.img_input = img_input
#         self.img_is_a_mask = img_is_a_mask
#         self.COMPATIBLE = False
#         self.img_input_format = False
#
#     def is_numpy_array(self, img):
#         return isinstance(img, np.ndarray)
#
#     def has_three_channels(self, img):
#         return img.ndim == 3 and img.shape[-1] == 3
#
#     def has_four_channels(self, img):
#         return img.ndim == 3 and img.shape[-1] == 4
#
#     def read_image(self):
#         pass
#
#     def load_img(self, path, COLORTRANSFORMATION):
#         temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#         temp = cv2.cvtColor(temp, COLORTRANSFORMATION)
#         return temp
#
#     def read_img_with_user_agent(self, url):
#         user_agent = {'user-agent': 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) ..'}
#         http = urllib3.PoolManager(10, headers=user_agent)
#         r1 = http.urlopen('GET', url)
#         LINK_IS_IMAGE = False
#         im2arr = []
#         try:
#             image = Image.open(io.BytesIO(r1.data))
#             im2arr = np.array(image)
#             LINK_IS_IMAGE = True
#         except UnidentifiedImageError:
#             pass
#         return LINK_IS_IMAGE, im2arr,
#
#     def is_path(self, s):
#         # Regular expression for validating file paths
#         path_regex = re.compile(
#             r'^(/|\\|[a-zA-Z]:\\|\.\\|..\\|./|../)'  # Starts with /, \, C:\, .\, ..\, ./, or ../
#             r'(?:(?:[^\\/:*?"<>|\r\n]+\\|[^\\/:*?"<>|\r\n]+/)*'  # Directory names
#             r'[^\\/:*?"<>|\r\n]*)$',  # Last part of the path
#             re.IGNORECASE)
#         return re.match(path_regex, s) is not None
#
#     def is_url(self, s):
#         # A simple regular expression for validating URLs
#         url_regex = re.compile(
#             r'^(https?://|ftp://)'  # HTTP, HTTPS, or FTP protocols
#             r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
#             r'localhost|'  # localhost
#             r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or IP
#             r'(?::\d+)?'  # optional port
#             r'(?:/?|[/?]\S+)$', re.IGNORECASE)
#         return re.match(url_regex, s) is not None
#
#     def is_base64(self, s):
#         """Check if the input string is Base64-encoded."""
#         try:
#             s = s.strip()
#             if len(s) % 4 != 0:
#                 return False
#             base64.b64decode(s, validate=True)
#             return True
#         except ValueError:
#             return False
#
#     def decode_img(self, encoded_img, mask=False):
#
#         if self.is_base64(encoded_img):
#             decoded_img = base64.b64decode(encoded_img)
#             img_np_arr = np.frombuffer(decoded_img, np.uint8)
#             if mask:
#                 img = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)
#                 if img is not None and len(img.shape) == 3 and img.shape[2] == 2:
#                     pass  # Not sure what 'pass' is intended for, possibly apply some processing here?
#                 else:
#                     img = img[:, :, 2]  # Assuming this intends to extract a specific channel, needs more clarification.
#             else:
#                 img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
#             return True, img
#         else:
#             return False, None
#
#     def encode_img(self, img):
#         _, img_buffer = cv2.imencode('.webp', img)
#         encoded_img = base64.b64encode(img_buffer)
#         # return encoded_img
#         return encoded_img.decode('utf-8')
#
#     def read_img_input(self):
#         if isinstance(self.img_input, str):
#             if self.is_url(self.img_input):
#                 LINK_IS_IMAGE, img = self.read_img_with_user_agent(self.img_input)
#                 if LINK_IS_IMAGE:
#                     if self.has_four_channels(img):
#                         img = img[:, :, :3]
#                         self.img = img
#                     elif self.has_three_channels(img):
#                         self.img = img
#                     if self.img_is_a_mask:
#                         self.img = self.img[:, :, 0]
#
#                     self.COMPATIBLE = True
#                 else:
#                     pass
#             else:
#                 is_img_input_a_base64_img, img = self.decode_img(self.img_input)
#                 if is_img_input_a_base64_img:
#                     if self.has_four_channels(img):
#                         img = img[:, :, :3]
#                         self.img = img
#                     elif self.has_three_channels(img):
#                         self.img = img
#                     if self.img_is_a_mask:
#                         self.img = self.img[:, :, 0]
#                     self.COMPATIBLE = True
#                 elif self.is_path(self.img_input):
#                      img = self.load_img(self.img_input, cv2.COLOR_BGR2RGB)
#                      if self.has_four_channels(img):
#                          img = img[:, :, :3]
#                          self.img = img
#                      elif self.has_three_channels(img):
#                          self.img = img
#                      if self.img_is_a_mask:
#                          self.img = self.img[:, :, 0]
#                      self.COMPATIBLE = True
#
#         elif isinstance(self.img_input, np.ndarray):
#             img=self.img_input
#             if self.has_four_channels(img):
#                 img = img[:, :, :3]
#                 self.img = img
#             elif self.has_three_channels(img):
#                 self.img = img
#             if self.img_is_a_mask:
#                 self.img = self.img[:, :, 0]
#             self.COMPATIBLE = True
#
#
