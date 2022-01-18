## Dataset Format

Images are present in ``` imgs ``` folder.  
Corresponding PNG annotations are present in ``` pngs ``` folder.

``` annotations.jsonl ``` contains the annotations for the images. The format of the annotations are:

```
{
    "img_fn": 
    "png_ann_fn": 
    "img_id": original image id from validation split of COCO-2017 
    "bboxes": [ {
            "category":
            "category_id":
            "bbox" : [x_min y_min x_max y_max]
        }
        .
        .
    ]
}
```

A sample of 10 images with their annotations are given. But at test time, the data will be different (the format will be the same so it will not affect the code)
