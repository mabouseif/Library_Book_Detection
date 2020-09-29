from detect import BookDetector
import skimage.io

if __name__ == '__main__':
    # Instantiate BookDetector object for segmentation
    book_detector = BookDetector()

    # Manually input destination directory if saving result images is required
    # book_detector.dest_dir = DEST_DIR

    # Image path
    image_path = 'test_images/IMG_20200412_215128.jpg'
    # Read image
    image = skimage.io.imread(image_path)
    # Run detection
    results, N = book_detector.detect(image)
    # Segment instances and save if required by populating the book_detector.dest_dir member variable
    book_detector.segment(image, results, image_path, all_at_once=False, show=True)

    print('Done!')