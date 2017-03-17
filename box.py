import cv2
import numpy as np
from collections import deque
from scipy.ndimage.measurements import label

class Box():
    """
    """
    def __init__(self, image, threshold=1, queue_len=5):
        self.image = image
        self.current_boxes = []
        self.threshold = threshold
        self.current_heatmap = []
        self.avg_heatmap = []
        self.heatmap_queue = deque(maxlen=queue_len)
        self.labels = None
        self.final_boxes = []

    def add_boxes(self, boxes):
        """
        Add all hot boxes (e.g. found cars)
        """
        self.current_boxes = boxes
        # find heatmap
        self.current_heatmap = self.find_heatmap()
        self.heatmap_queue.append(self.current_heatmap)
        self.avg_heatmap = np.mean(self.heatmap_queue, axis=0)
        # get labels out of heatmap
        self.labels = label(self.current_heatmap)
        # get final boxes out of labels
        self.final_boxes = self.find_final_boxes()

    def add_heat(self):
        """
        Add heat (+1) for all pixels within all hot boxes
        """
        heat = np.zeros_like(self.image[:, :, 0]).astype(np.float)
        # Iterate through list of bboxes
        for box in self.current_boxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heat

    def apply_threshold(self, heat):
        """
        Only enable heat boxes above threshold
        """
        # Zero out pixels below the threshold
        heat[heat <= self.threshold] = 0
        # Return thresholded map
        return heat

    def find_heatmap(self):
        """
        Find heatmap image
        """
        # Add heat to each box in box list
        heat = self.add_heat()
        # Apply threshold to help remove false positives
        #heat = self.apply_threshold(heat)
        # Visualize the heatmap when displaying
        return np.clip(heat, 0, 255)

    def find_final_boxes(self):
        """
        Use scipy label function and heatmap to identify objects
        """
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
        """
        Draw final rectangles around found objects
        """
        draw_img = np.copy(img)
        for bbox in self.final_boxes:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return draw_img

    def draw_hot(self, img):
        """
        Draw hot boxes rectangles around found objects
        """
        draw_img = np.copy(img)
        for bbox in self.current_boxes:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (255, 0, 0), 2)
        # Return the image
        return draw_img
