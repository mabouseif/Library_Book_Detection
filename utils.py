import json
import base64
import io
import numpy as np
import math
import PIL
from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import scipy.misc
import glob, time, pickle, sys, random, copy
# from planar import BoundingBox
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean, rotate
# import png


# Stolen from Kentaro Wada
# https://github.com/wkentaro/labelme/blob/739042f74dac497cc607faa4773ef7c7dbc67f84/labelme/utils/image.py#L17
# https://github.com/wkentaro/labelme/blob/master/labelme/cli/json_to_dataset.py
def img_data_to_arr(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

# Stolen from Kentaro Wada
# https://github.com/wkentaro/labelme/blob/739042f74dac497cc607faa4773ef7c7dbc67f84/labelme/utils/image.py#L17
# https://github.com/wkentaro/labelme/blob/master/labelme/cli/json_to_dataset.py
def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


def json_to_img(file_path_json, gray=False):
    with open(file_path_json, encoding='cp1252') as json_file: # Encoding utf-8 or cp1252??
        if len(json_file.readlines()) > 0:
            json_file.seek(0)
            data = json.loads(json_file.read())
        else:
            print('File empty')
            print(file_path_json)
            return None
            
    imageData = data.get('imageData')
    img = img_b64_to_arr(imageData)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    shapes = data.get('shapes')
    points = [polygon['points'] for polygon in shapes]
    
    return [img, points]

def plot_polygon_patches(points, ax):
    patches = []
    for bbox in points:
        patches.append(Polygon(bbox, closed=True))
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    ax.add_collection(p)
    
    
def get_bbox(polygon):
    bbox = BoundingBox([polygon[i] for i in range(len(polygon))])
    x1 = int(bbox.min_point[0])
    y1 = int(bbox.min_point[1])
    x2 = int(bbox.max_point[0])
    y2 = int(bbox.max_point[1])
    return x1, y1, x2, y2


def fix_bounds(value, max_val):
    if value > max_val:
        return max_val
    if value < 0:
        return 0
    return value

def get_bbox_new(img, polygon):
    max_width, max_height = img.size
    bbox = BoundingBox([polygon[i] for i in range(len(polygon))])
    x1 = fix_bounds(int(bbox.min_point[0]), max_width)
    y1 = fix_bounds(int(bbox.min_point[1]), max_height)
    x2 = fix_bounds(int(bbox.max_point[0]), max_width)
    y2 = fix_bounds(int(bbox.max_point[1]), max_height)
    return x1, y1, x2, y2


def point_transform(point, scale, radians, trans, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = (ox + math.cos(radians) * (x*scale - ox) + math.sin(radians) * (y*scale - oy)) + trans[0]
    qy = (oy + -math.sin(radians) * (x*scale - ox) + math.cos(radians) * (y*scale - oy)) + trans[1]

    return qx, qy



def transform_img_and_points(img, points, scale, angle_deg):
    
    angle_rad = angle_deg * np.pi/180.0
    
    new_height = int(img.shape[0] * scale)
    new_width = int(img.shape[1] * scale)
    
    img_resized = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    point_rotate_about = np.array([img_resized.shape[1]/2.0, img_resized.shape[0]/2.0]) # x, y point
    
    img_rotated = rotate(img_resized, angle_deg, resize=True, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=True).astype(np.uint8)
    shift_by = np.array([(img_rotated.shape[1]-img_resized.shape[1])/2.0, (img_rotated.shape[0]-img_resized.shape[0])/2.0]) # shift in y, shift in x
    
    new_points = [[point_transform(point, scale=scale, radians=angle_rad, trans=shift_by, origin=point_rotate_about) for point in bbox]for bbox in points]

#     points = [np.dot(bbox, affine_matrix) for bbox in points]
    
    return img_rotated, new_points