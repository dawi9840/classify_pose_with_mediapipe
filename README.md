# classify_pose_with_mediapipe
Use mediapipe holistic model to classify pose application.  

![cat_camel_out](https://user-images.githubusercontent.com/19554347/129997232-cf2d084e-b8d0-417b-9885-b2895689bee6.gif)   

This repository use [mediapipe](https://github.com/google/mediapipe) to get the body pose of 33 key points (coordinate points) which create a dataset to be classified the body pose.  

The idea is from [AI Pose Estimation with Python and MediaPipe | Plus AI Gym Tracker Project](https://youtu.be/06TE_U21FK4).   

I create 3 categories of body pose with a  example. You can add new body pose category for yourself by **pose_dataset_create.py**.   

# The file description

**pose_dataset_create.py** - Check the csv dataset is exist or not and with it to add new pose category.    

**model_train.py** - Use pose dataset to train classifier model and save model weights.

**model_test.py** - Input an image to test the model predict of pose category.   

**demo_classify_pose.py** - Display pose classify and save result with a input video.

## Install

**Conda Virtualenv**

```bash

conda create --name [env_name]  python=3.7
conda activate [env_name]
pip install -r requirements.txt

```

## Usage

**demo_classify_pose.py**

```bash

python demo_classify_pose.py

```

If want to use camera demo just modify video_path set to 0 (or your cameras position number).   



