# Face Detection

## Methods

### 1. RetinaFace

Based on pytorch.

Can use Mobilenet or Resnet50(recommended for accuracy) as backbone.

The trained model weights are in weights/retina_face folder.

Note: Resnet50 model is bigger than 100Mb, can be downloaded
via https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1

result/result_100.csv : is_face-labels for First 100 images.

result/result_all.csv : result of all images in /profile pics.

results/result_User_demo_profiles_json.csv : result for images in result_User_demo_profiles_json, if the image is not
existed, the coefficient is set to -1 (totally 85 items). And 807 items are detected as no-face.

#### Problem:

A few cartoon person are detected as human face.

### 2. SCRFD

Based on OpenCV 4.0+

**##TO-DO**