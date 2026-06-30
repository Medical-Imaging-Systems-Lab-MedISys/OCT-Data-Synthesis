import gdown
import requests
import os
from natsort import natsorted
import cv2
import numpy as np
import pathlib
import pandas as pd
import shutil
import imageio
### Use gdown to download the file ###
### If not installed yet use: 'pip install gdown' ###

gdown.download("https://drive.google.com/u/0/uc?id=1Rv82F7CjPveyONdy1YbRHh05emCb6_Eu&export=download")
### Use gdown to download the old version of the dataset ###

url = "http://cutt.ly/wGOhQeK"
session = requests.Session()
resp = session.head(url, allow_redirects=True)
link = "https://drive.google.com/u/0/uc?id="+resp.url.split("/")[5]+"&export=download";

gdown.download(link)
### Use unrar to extract the downloaded file ###
### On Ubuntu e.g. use 'sudo apt install unrar' ###

import os; os.system('unrar x "Macular Dataset-Heidelberg.rar" > Output_Log.out')
import os; os.system('rm -rf "Macular Dataset-Heidelberg.rar"')
### Use unrar to extract the downloaded file ###
### On Ubuntu e.g. use 'sudo apt install unrar' ###

import os; os.system('unrar x "Macular-Dataset-R.Rasti_old.rar" > Output_Log.out')
import os; os.system('rm -rf "Macular-Dataset-R.Rasti_old.rar"')
### Get the password from "https://sites.google.com/site/hosseinrabbanikhorasgani/available-datasets/dataset-for-oct-classification-50-normal-48-amd-50-dme"
import os; os.system('mkdir Macular-Dataset-R.Rasti_old')
import os; os.system('unzip -q -P "typepasswordhere" "Macular-Dataset-R.Rasti-part01.zip" -d "./Macular-Dataset-R.Rasti_old/"')
import os; os.system('unzip -q -P "typepasswordhere" "Macular-Dataset-R.Rasti-part02.zip" -d "./Macular-Dataset-R.Rasti_old/"')
import os; os.system('unzip -q -P "typepasswordhere" "Macular-Dataset-R.Rasti-part03.zip" -d "./Macular-Dataset-R.Rasti_old/"')
import os; os.system('unzip -q -P "typepasswordhere" "Macular-Dataset-R.Rasti-part04.zip" -d "./Macular-Dataset-R.Rasti_old/"')
import os; os.system('rm -rf "Macular-Dataset-R.Rasti-part01.zip" "Macular-Dataset-R.Rasti-part02.zip" "Macular-Dataset-R.Rasti-part03.zip" "Macular-Dataset-R.Rasti-part04.zip"')
### Download Kermany dataset ###
#!wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/rscbjbr9sj-3.zip

import os; os.system('wget https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded -O ZhangLabData.zip -o Output_Log.out')
### Use unzip to extract the downloaded file ###

import os; os.system('unzip -q ZhangLabData.zip')
import os; os.system('rm -rf ZhangLabData.zip')
def make_target_dirs(target_paths):
    for dirname in set(os.path.dirname(p) for p in target_paths):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
df = pd.read_csv("./paths/original_paths.csv",header=None)
toPath = list(df[0]);
fromPath = list(df[1]);

for i in range(0, len(fromPath)):
    fromImage = np.array(imageio.imread(fromPath[i]));
    
    make_target_dirs([toPath[i]]);
    
    if(".jpeg" in toPath[i]):
        shutil.copy2(fromPath[i], toPath[i]);
    else:
        imageio.imsave(toPath[i], fromImage);
    
    toImage = np.array(imageio.imread(toPath[i]));
    if(np.array_equal(fromImage, toImage) == False):
        print (i)
        print (fromImage);
        print (toImage);
        #break;
df = pd.read_csv("./paths/manual_paths.csv",header=None)
toPath = list(df[0]);
fromPath = list(df[1]);

for i in range(0, len(fromPath)):
    fromImage = np.array(imageio.imread(fromPath[i]));
    
    make_target_dirs([toPath[i]]);
    
    fromImage = cv2.resize(fromImage,dsize=(512,512));
    imageio.imsave(toPath[i], fromImage);
    
    toImage = np.array(imageio.imread(toPath[i]));
    
    if(np.array_equal(fromImage, toImage) == False):
        print (i)
        print (fromImage);
        print (toImage);
        #break;
df = pd.read_csv("./paths/automatic_paths.csv",header=None)
toPath = list(df[0]);
fromPath = list(df[1]);

for i in range(0, len(fromPath)):
    fromImage = np.array(imageio.imread(fromPath[i]));
    
    if(".jpeg" in toPath[i]):
        toPath[i] = toPath[i].replace(".jpeg", ".png");
        
    make_target_dirs([toPath[i]]);
    
    fromImage = cv2.resize(fromImage,dsize=(512,512));
    imageio.imsave(toPath[i], fromImage);
    
    toImage = np.array(imageio.imread(toPath[i]));
    
    if(np.array_equal(fromImage, toImage) == False):
        print (i)
        print (fromImage);
        print (toImage);
        #break;
df = pd.read_csv("./paths/detection_paths.csv",header=None)
toPath = list(df[0]);
fromPath = list(df[1]);

for i in range(0, len(fromPath)):
    fromImage = np.array(imageio.imread(fromPath[i]));
    
    make_target_dirs([toPath[i]]);
    
    fromImage = cv2.resize(fromImage,dsize=(512,512));
    
    toPath[i] = toPath[i].replace(".jpeg", ".png");
    
    imageio.imsave(toPath[i], fromImage);
    
    toImage = np.array(imageio.imread(toPath[i]));
    if(np.array_equal(fromImage, toImage) == False):
        print (i)
        #print (fromImage);
        #print (toImage);
        #break;
# Remove downloaded images if not needed
import os; os.system('rm -rf CellData')
import os; os.system('rm -rf Dataset_3x50_Final')
import os; os.system('rm -rf Macular-Dataset-R.Rasti_old')
import os; os.system('rm -rf Output_Log.out')

