import os
import sys
import glob
import random
import math
import warnings
import numpy as np
import skimage.io
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import BOOK config
# sys.path.append(os.path.join(ROOT_DIR, "book/"))  # To find local version
import book


# Ignore depracation warnings
tf.logging.set_verbosity(tf.logging.ERROR)



class BookDetector():
    def __init__(self, root_dir="./Mask_RCNN", model_path="models/mask_rcnn_book_0999.h5"):

        # Root directory of the project
        ROOT_DIR = os.path.abspath(root_dir)

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library

        # Local path to trained weights file
        # BOOK_MODEL_PATH = os.path.join(ROOT_DIR, model_path) # REMEMBER TO CHANGE THIS
        BOOK_MODEL_PATH = "./models/mask_rcnn_book_0999.h5"
        
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        
        # Load config
        config = book.BookConfig()
        # config.display()

        # Destination directory
        self.dest_dir = None

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(BOOK_MODEL_PATH, by_name=True)

    # Check presence of detections
    def num_instances_check(self, results):
        # Number of instances
        r = results
        boxes = r['rois']
        masks = r['masks']
        class_ids = r['class_ids']
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            exit()
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        return N

    # Model forward pass
    def detect(self, image):
        # Run detection
        results = self.model.detect([image], verbose=1)
        N = self.num_instances_check(results[0])

        return results, N

    # TODO
    def visualize(self, results):
        # Visualize results
        r = results[0]
        class_names = ['BG', 'book']
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])


    # Saves images at dest_dir
    def save_image(self, image, original_file_path, instance_idx=0):
        dot_idx = original_file_path.rfind('.')
        slash_idx = original_file_path.rfind('/')
        if slash_idx == -1:
            tmp_name = original_file_path[:dot_idx]
        else:
            tmp_name = original_file_path[slash_idx+1:dot_idx]
        extension = original_file_path[dot_idx:]

        if instance_idx:
            new_file_path = os.path.join(self.dest_dir, tmp_name + '_' + str(instance_idx) + extension)
        else:
            new_file_path = os.path.join(self.dest_dir, tmp_name + '_all' + extension)

        image = Image.fromarray(image)
        image.save(new_file_path)
        print('Image {} saved as {}'.format(original_file_path, new_file_path))

    # Segment detections, show, and save if required
    def segment(self, image, results, image_path, all_at_once=True, show=False):
        """
        image: Input 3-channel image
        results: The output/detections from detect function
        all_at_once: If True, then ALL detected instances are returned in one image
                     If False, EACH detected instance is returned in an individual image
        """
        r = results[0]
        N = self.num_instances_check(r)
        mask_img = np.zeros_like(image)
        stitched = np.zeros((image.shape[0]*2, image.shape[1], image.shape[2]), dtype=np.uint8)

        for i in range(r['masks'].shape[-1]):
            mask_img[[r['masks'][:, :, i]]] = image[[r['masks'][:, :, i]]]
            stitched[0:image.shape[0], :, :] = image
            stitched[image.shape[0]:, :, :] = mask_img
            if not all_at_once:
                if self.dest_dir:
                    self.save_image(mask_img, image_path, instance_idx=i+1)
                mask_img = np.zeros_like(image)
                if show:
                    plt.imshow(stitched)
                    plt.show()
                    plt.axis('off')

        if all_at_once:
            if self.dest_dir:
                self.save_image(mask_img, image_path)
            if show:
                plt.imshow(stitched)
                plt.show()
                plt.axis('off')



if __name__ == '__main__':

    # Adding arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--all_at_once', dest='all_at_once', action='store_true', help='If True, then ALL detected instances are returned in one image. \
                                                                                        If False, EACH detected instance is returned in an individual image')
    parser.add_argument('--show', dest='show', action='store_true', help='If provided, results will be shown')
    parser.add_argument('--image_path', dest='image_path', action='store', type=str, help='Path to image file')
    parser.add_argument('--image_dir', dest='image_dir', action='store', type=str, help='Directory of images')
    parser.add_argument('--dest_dir', dest='dest_dir', action='store', type=str, help='If provided, destination directory where output will be saved')
    

    # Argument parsing
    args = parser.parse_args()

    image_path = args.image_path
    image_dir = args.image_dir
    dest_dir = args.dest_dir
    all_at_once = args.all_at_once
    show = args.show

    # Input exist check
    if not (image_path or image_dir):
        print('An image path or an image directory must be provided!')
        exit()
    
    # Destination dir check
    if dest_dir and not os.path.isdir(dest_dir):
        print('Destination directory is invalid!')
        exit()

    # Instantiate BookDetector object for segmentation
    book_detector = BookDetector()
    book_detector.dest_dir = dest_dir


    # Note: The following snippet could be merged as one, but then 2 sequential
    # loops would be needed

    # Single image input case 
    if image_path and os.path.isfile(image_path):
        image = skimage.io.imread(image_path)
        if len(image.shape) == 1:
            image = image.reshape((image.shape[0], image.shape[1], 1))
            image = np.concatenate((image, image, image), axis=2)
        assert image.shape[2] == 3, "Image does not have 3-channels!"
        
        results, N = book_detector.detect(image)
        book_detector.segment(image, results, image_path, all_at_once=all_at_once, show=show)

    # Directory input case
    elif image_dir and os.path.isdir(image_dir):
        for file_path in glob.glob(os.path.join(image_dir, '*')):
            image = skimage.io.imread(file_path)
            if len(image.shape) == 2:
                image = image.reshape((image.shape[0], image.shape[1], 1))
                image = np.concatenate((image, image, image), axis=2)
            assert image.shape[2] == 3, "Image does not have 3-channels!"

            results, N = book_detector.detect(image)
            book_detector.segment(image, results, file_path, all_at_once=all_at_once, show=show)

    else:
        print('Not a file or directory')
        exit()

    print('Done!')

