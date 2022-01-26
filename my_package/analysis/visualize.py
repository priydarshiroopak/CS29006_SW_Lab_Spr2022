#Imports
from PIL import ImageDraw, Image, ImageFont
import numpy as np

def plot_visualization(img, bboxes, labels, output):  # Write the required arguments
    img = Image.fromarray(np.uint8(img.transpose((1, 2, 0)) * 255))
    for i in range(min(5, len(bboxes))):
        ImageDraw.Draw(img).rectangle(bboxes[i], outline='red', width=4)
    for i in range(min(5, len(bboxes))):
        ImageDraw.Draw(img).text(bboxes[i][0], labels[i], fill='yellow')
    img.save(output)

  # The function should plot the predicted segmentation maps and the bounding boxes on the images and save them.
  # Tip: keep the dimensions of the output image less than 800 to avoid RAM crashes.