# classify_pose_with_mediapipe
Use mediapipe holistic model to classify pose application.  

![cat_camel_out](https://user-images.githubusercontent.com/19554347/129997232-cf2d084e-b8d0-417b-9885-b2895689bee6.gif)   

This repo use [mediapipe](https://github.com/google/mediapipe) to get the body pose of 33 key points which create a dataset to be classified the body pose.  

The idea is from [AI Pose Estimation with Python and MediaPipe | Plus AI Gym Tracker Project](https://youtu.be/06TE_U21FK4).   

I create 3 pose class to classify with a example. You can addd new pose class yourself in **pose_dataset_create.py**.

# The file description

**pose_dataset_create.py** - Check the data is exit or not and create dataset with new pose class.   

**model_train.py** - Use pose dataset to train Classifier model.

**model_test.py** - Input a image to test model pridect pose.

**demo_classify_pose.py** - Display input video with classify result video.



