#Imports
from pickletools import uint8
from my_package.model import InstanceSegmentationModel
from my_package.data import Dataset
from my_package.analysis import plot_visualization
from my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def experiment(annotation_file, segmentor, transforms, outputs):
    '''
        Function to perform the desired experiments

        Arguments:
        annotation_file: Path to annotation file
        segmentor: The image segmentor
        transforms: List of transformation classes
        outputs: path of the output folder to store the images
    '''

    #Create the instance of the dataset.
    data = Dataset(annotation_file, transforms)


    #Iterate over all data items.
    for i in range(len(data)):
        pred_boxes, pred_masks, pred_class, pred_score = segmentor(data[i]['image'])
        plot_visualization(data[i]['image'], pred_masks[:3], pred_boxes[:3], pred_class[:3], outputs + 'imgs/{}.jpg'.format(i))


    #Get the predictions from the segmentor.
    #Draw the segmentation maps on the image and save them.
    i = int(input("Enter number of image you want to perform analysis experiments on (0-9): "))
    while(i<0 or i>9):
        i = int(input("Enter valid number! (0-9): "))
        
    _, im_h, im_w = data[i]['image'].shape
    myanalysis = {'original': ('Original Image', []),
                  'flip': ('Horizontally Flipped', [FlipImage()]),
                  'blur': ('Blurred', [BlurImage(4)]),
                  'twice': ('Twice Rescaled', [RescaleImage((2 * im_w, 2 * im_h))]),
                  'half': ('Half Rescaled', [RescaleImage((int(im_w / 2), int(im_h / 2)))]),
                  'right': ('90 Degree Right Rotated', [RotateImage(-90)]),
                  '45deg': ('45 Degree Left Rotated', [RotateImage(45)])}

    #Do the required analysis experiments.

    for ind, item in enumerate(myanalysis.items()):
        key, val = item
        data.transforms = val[1]
        pred_boxes, pred_masks, pred_class, pred_score = segmentor(data[i]['image'])
        plot_visualization(data[i]['image'], pred_masks[:3], pred_boxes[:3], pred_class[:3], outputs + 'img_analysis/{}.jpg'.format(key))
        plt.subplot(2, 4, ind + 1, title=val[0])
        plt.imshow(Image.open(outputs + 'img_analysis/{}.jpg'.format(key)))
    
    plt.savefig(outputs+'plotted image.jpg', bbox_inches='tight')
    plt.show()
    


def main():
    segmentor = InstanceSegmentationModel()
    experiment('./data/annotations.jsonl', segmentor, [], 'output/') # Sample arguments to call experiment()


if __name__ == '__main__':
    main()
