#  library_book_detection_project

Book spine detection using Mask-RCNN


### Prerequisites
```
Python 3.4
TensorFlow 1.15
Keras 2.1.0
```
and other common packages listed in requirements.txt.

### Installing

Run `setup_script.sh`. It is also commented in case of wanting to install line by line manually.
Otherwise, follow the following steps:

1. Clone this repository

2. Install dependencies
```
pip install tensorflow==1.15.0
pip install -r Mask_RCNN/requirements.txt
```

3. Run setup from the Mask_RCNN repository root directory


```
python setup.py install
```

## Getting Started

1. Download the [Model](https://drive.google.com/open?id=14xFenBvF8uIyG8t2U8w2dcWfXkw-ibtF) and put in `models` directory.
2. You can either run inference for a single image or a directory using the `--image_path` or `--image_dir` arguments, respectively. If the `--dest_dir`argument is provided, the results will be saved provided path. If the `--all_in_one` flag is provided, *all* segmented instances will be output in the same image, otherwise *each* segmented instance will be output in a separate image. If `--show` flag is provided, segmentation results will be shown.
```
python detect.py --all_at_once --image_path IMAGE_PATH --image_dir IMAGE_DIR --output_dir OUTPUT_DIR
```
To see all options, run `python detect.py -h`:
```
usage: detect.py [-h] [--all_at_once] [--show] [--image_path IMAGE_PATH]
                 [--image_dir IMAGE_DIR] [--dest_dir DEST_DIR]
```

## Augmentation
The script `augment.py` is provided to augment the dataset with simple affine transformations (rotatio, scaling, etc..).
To see all options, run `python augment.py -h`: 
```
usage: augment.py [-h] --dir_json DIR_JSON --dir_out DIR_OUT --dir_points_out
                  DIR_POINTS_OUT [--gray] [--scale SCALE]
                  [--degree_increments DEGREE_INCREMENTS]

optional arguments:
  -h, --help            show this help message and exit
  --dir_json DIR_JSON   Directory of dataset in JSON format
  --dir_out DIR_OUT     Directory of output augmented dataset
  --dir_points_out DIR_POINTS_OUT
                        Directory of output pickle file that contains the
                        modified annotations
  --gray                Convert images to gray scale
  --scale SCALE         Scaling transform of the augmentations
  --degree_increments DEGREE_INCREMENTS
                        If provided Rotate images in increments of
                        degree_increments till 360, otherwise equates to
                        covnerting images from JSON to png
```


## Example
By running the following example, detections will be done on the test_images provided. Should also be tried *without* the `--all_at_once` flag. Provide DEST_DIR and run:
```
python detect.py --all_at_once --show --image_dir test_images/ --dest_dir DEST_DIR
```

A simplified demo version on how to peform detections in own code can be seen in `demo.py`. Run using:
```
python demo.py
```

## Results
Results on photos taken using phone camera: 

<img src="assets/1.png " width="210">
<img src="assets/2.png " width="210">
<img src="assets/3.png " width="210">
<img src="assets/4.png " width="210">
<br/><br/>
<img src="assets/all.png " width="400">
<!-- ![Instance Segmentation Sample](assets/all.png | width=100) -->


# TODO
1. Run inference on GPU.
2. Calculate metrics.
3. Train from scratch on both COCO dataset and the book spine dataset, either simultaneously or sequentially, with more suitable anchors for the dataset in order to improve detection for tilted books and images with planes not parallel to shelf.
4. Make sure no problems reading images from directory when there's another directory within.
5. Add grayscale conversion to `augment.py`
6. Extract midpoint of book spine.
