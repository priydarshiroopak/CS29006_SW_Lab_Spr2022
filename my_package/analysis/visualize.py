#Imports
from PIL import ImageDraw, Image #, ImageFont
import numpy as np

def plot_mask(img, mask, i):    #function to plot mask(s) on the image
    mask = np.array(Image.fromarray(np.uint8(mask*255)).convert("RGBA"))
    #initialising and setting multiplier array for transparency and color of masks
    multiplier = [0, 0, 0, 150]
    multiplier[i%3] = 1
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if (mask[y, x][0]==0).all():
                mask[y, x][3] = 0
            else:
                mask[y,x]=mask[y,x]*multiplier
    mask = Image.fromarray(mask)
    img.paste(mask, box=(0,0), mask= mask)

def plot_visualization(img, masks, bboxes, labels, output):  # Write the required arguments
    img = Image.fromarray(np.uint8(img.transpose((1, 2, 0)) * 255))
    for i, box in enumerate(bboxes):
        plot_mask(img, masks[i][0], i)
        ImageDraw.Draw(img).rectangle(box, outline='turquoise', width=3)
        ImageDraw.Draw(img).text([bboxes[i][0][0]+3,bboxes[i][0][1]+2], labels[i], fill='palegreen')
    img.save(output)

  # The function should plot the predicted segmentation maps and the bounding boxes on the images and save them.
  # Tip: keep the dimensions of the output image less than 800 to avoid RAM crashes.