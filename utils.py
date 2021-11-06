import pandas as pd
import numpy as np
import csv
import os

# extract img_path from json and join with csv file on user id
def read_files(json_file, csv_file, img_dir):
    json_with_image_path = pd.read_json(json_file)
    csv = pd.read_csv(csv_file)
    joined_df = csv.join(json_with_image_path.set_index('id'), on = 'user_id',how='inner').drop(['name','screen_name','description','lang'],axis = 1).dropna(axis= 0)

    img_filenames = joined_df['img_path'].to_list()
    for idx, path in enumerate(img_filenames):
        #print(img_dir+'/'+str(path.split('/')[-1]))
        img_filenames[idx]= img_dir+path.split('/')[-1]
        
    label = joined_df['race'].to_list()
    return img_filenames, label

# Helper function to detect faces in the list
def detect_face(img):
    # TODO: detect if img (converted numpy array format) is a face
    pass

if __name__ == '__main__':
    # test paths
    img_lst, label_lst = read_files('./demographicPrediction/User demo profiles.json', './demographicPrediction/labeled_users.csv','./demographicPrediction/profile_pics/profile pics/')
    print(img_lst[:3])
    

