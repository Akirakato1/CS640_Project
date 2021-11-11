#Face Detection

##Methods

###1. RetinaFace

Based on pytorch.

Can use Mobilenet or Resnet50(recommended for accuracy) as backbone.

The trained model weights are in weights/retina_face folder.

Note: Resnet50 model is bigger than 100Mb, can be downloaded via https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1

Have generated is_face-labels for First 100 images, 

the result is in result/result_100.csv.

The result of all images in /profile pics will be updated by the noon of Nov.13 2021.

####Problem:

A few cartoon person are detected as human face.

###2. SCRFD

Based on OpenCV 4.0+

**##TO-DO**