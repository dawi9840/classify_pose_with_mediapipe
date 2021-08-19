from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp 

def img_model_test(image, model):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Recolor Feed
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )    

        # Extract Pose landmarks
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        # Concate rows
        row = pose_row

        # Make Detections
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        print(f'class: {body_language_class}, prob: {body_language_prob}')

        return body_language_class, body_language_prob

if __name__ == '__main__':

    img = cv2.imread('./resource/imgs/test1.png')
    model_weights = 'body_language3.pkl'
    
    # Load Model.
    with open(model_weights, 'rb') as f:
        model = pickle.load(f)

    # Input image to test model predict. 
    img_model_test(image=img, model=model)