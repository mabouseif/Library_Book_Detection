import argparse
import glob, time, pickle
from PIL import Image
from utils import transform_img_and_points, json_to_img
import os, sys
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_json', required=True, help='Directory of dataset in JSON format')
    parser.add_argument('--dir_out', required=True, help='Directory of output augmented dataset')
    parser.add_argument('--dir_points_out', required=True, help='Directory of output pickle file that contains the modified annotations')
    parser.add_argument('--gray', dest='gray', action='store_true', help='Convert images to gray scale')
    parser.add_argument('--scale', dest='scale', default=1.0, type=float, action='store', help='Scaling transform of the augmentations')
    parser.add_argument('--degree_increments', dest='degree_increments', default=360, type=int, action='store', help='If provided Rotate images in increments of degree_increments till 360, \
                                                                                                                      otherwise equates to covnerting images from JSON to png')

    args = parser.parse_args()

    dir_json = args.dir_json
    dir_out = args.dir_out
    dir_points_out = args.dir_points_out
    gray = args.gray
    scale = args.scale
    degree_increments = args.degree_increments

    if not (os.path.isdir(dir_json) and os.path.isdir(dir_out) and os.path.isdir(dir_points_out)):
        print("Directory invalid!")
        exit()

    points_dict_new = {}
    count = 0

    start_time = time.time()
    for file_path in glob.glob(os.path.join(dir_json, '*')):
        img_and_points = json_to_img(file_path, gray=False)
        if img_and_points is not None:
            k = file_path[-24:-5]
            print('\rImage number: {} and key: {}'.format(count+1, k), end="")
            sys.stdout.flush()
            
            img = img_and_points[0]
            points = img_and_points[1]
            
            for angle_deg in range(0, 360, degree_increments):
                k = file_path[-24:-5] + '_' + str(angle_deg)
                
                image_transformed, new_points = transform_img_and_points(img, points, scale, angle_deg)
                
                new_height = int(image_transformed.shape[0] * scale)
                new_width = int(image_transformed.shape[1] * scale)
                new_points = [list([list(y) for y in x]) for x in new_points]
                points_dict_new[k] = [new_points, new_height, new_width]

                image_transformed = Image.fromarray(image_transformed)
                new_file_path = os.path.join(dir_out, k + '.png')
                image_transformed.save(new_file_path)
            
        count += 1
        
    end_time = np.round((time.time() - start_time) / 60.0, 2)
    print('\nElapsed: {} minutes'.format(end_time))

    # Save points dict
    f = open(os.path.join(dir_points_out, 'points.pkl'),"wb")
    pickle.dump(points_dict_new,f)
    f.close()
