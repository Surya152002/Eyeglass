import cv2
import numpy as np
from segmentation_model import SegmentationModel

def post_process_segmentation(segmented_image):
    segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    smoothed_image = cv2.morphologyEx(segmented_gray, cv2.MORPH_CLOSE, kernel)
    _, thresholded_image = cv2.threshold(smoothed_image, 200, 255, cv2.THRESH_BINARY)
    result_image = cv2.bitwise_and(segmented_image, segmented_image, mask=thresholded_image)
    return result_image

# Load and preprocess the image
image = cv2.imread('eyeglass_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize and load the segmentation model
model = SegmentationModel()
model.load_model('segmentation_model_weights.h5')

# Perform segmentation on the image
segmented_image = model.segment_image(image)

# Post-process the segmented image for smooth edges and white background
processed_image = post_process_segmentation(segmented_image)

# Display the processed image
cv2.imshow('Processed Eyeglass Segmentation', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
