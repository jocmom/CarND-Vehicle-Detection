import cv2
import numpy as np
from scipy.ndimage.measurements import label

class Box():
    """
    """
    def __init__(self, image, hot_boxes, threshold=1):
        self.image = image
        self.hot = hot_boxes
        self.threshold = threshold
        # find heatmap
        self.heatmap = self.find_heatmap()
        # get labels out of heatmap
        self.labels = label(self.heatmap)
        # get final boxes out of labels
        self.final = self.find_final_boxes()

    def add_heat(self):
        heat = np.zeros_like(self.image[:, :, 0]).astype(np.float)
        # Iterate through list of bboxes
        for box in self.hot:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heat

    def apply_threshold(self, heat):
        # Zero out pixels below the threshold
        heat[heat <= self.threshold] = 0
        # Return thresholded map
        return heat

    def find_heatmap(self):
        # Add heat to each box in box list
        heat = self.add_heat()
        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat)
        # Visualize the heatmap when displaying
        return np.clip(heat, 0, 255)

    def find_final_boxes(self):
        final_boxes = []
        for l in range(1, self.labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == l).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            final_boxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
        return final_boxes

    def draw_final(self, img):
        for bbox in self.final:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img
