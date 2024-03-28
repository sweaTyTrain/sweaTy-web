from django.db import models
import mediapipe as mp
import math
import cv2
import numpy as np
import tensorflow as tf
import joblib
import os



class TestModel(models.Model):
    label_0 = models.CharField(max_length=100, default='default_value')
    label_1 = models.CharField(max_length=100, default='default_value')
    label_2 = models.CharField(max_length=100, default='default_value')
    label_3 = models.CharField(max_length=100, default='default_value')
    label_4 = models.CharField(max_length=100, default='default_value')
    label_5 = models.CharField(max_length=100, default='default_value')
    label_6 = models.CharField(max_length=100, default='default_value')
    squatCnt = models.CharField(max_length=5, default='0')
    squatBeforeState = models.CharField(max_length=5, default='1')
    squatNowState = models.CharField(max_length=5, default='1')
    squatState = models.CharField(max_length=200, default='')
    squatAccuracy = models.CharField(max_length=10, default='-1')
    stateQueue = models.CharField(max_length=50, default='-1,-1,-1,-1,-1')
    classIdx = models.CharField(max_length=50, default='-1')



# 현재 스크립트의 디렉토리 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 모델 파일의 경로
MODEL_PATH = os.path.join(BASE_DIR, 'AImodel', 'gnb_model_3D_angle_noDis.joblib')

# 모델 확장자가 .h5일 때와 .joblib일 때 모델을 불러오는 방법이 다름
file_extension = MODEL_PATH.split('.')[-1]

if file_extension == 'h5':
    model = tf.keras.models.load_model(MODEL_PATH)
elif file_extension == 'joblib':
    model = joblib.load(MODEL_PATH)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# 3D 각도 계산 함수(객체)
def calculateAngle3D(landmark1, landmark2, landmark3):
    # landmark1에서 landmark2로 향하는 3차원 벡터 계산
    vector1 = np.array(landmark1) - np.array(landmark2)
    # landmark3에서 landmark2로 향하는 3차원 벡터 계산
    vector2 = np.array(landmark3) - np.array(landmark2)

    # 벡터의 크기를 고려하지 않기 위해 단위벡터로 환산
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)

    # 두 단위벡터의 내적을 구한다.(내적: 두 벡터의 방향 유사도(-1~1))
    dot_product = np.dot(unit_vector1, unit_vector2)

    # 내적을 바탕으로 두 단위벡터 사이의 각을 구한다.(라디안)
    angle_radians = math.acos(np.clip(dot_product, -1.0, 1.0))

    # 라디안 각을 degree로 변환
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees



# 3D 각도 계산 함수(좌표)
def calculateAngle3D_2(landmarkx1, landmarky1, landmarkz1, landmarkx2, landmarky2, landmarkz2, landmarkx3, landmarky3, landmarkz3):
    vector1 = np.array([landmarkx1, landmarky1, landmarkz1]) - np.array([landmarkx2, landmarky2, landmarkz2])
    # landmark3에서 landmark2로 향하는 3차원 벡터 계산
    vector2 = np.array([landmarkx3, landmarky3, landmarkz3]) - np.array([landmarkx2, landmarky2, landmarkz2])

    # 벡터의 크기를 고려하지 않기 위해 단위벡터로 환산
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)

    # 두 단위벡터의 내적을 구한다.(내적: 두 벡터의 방향 유사도(-1~1))
    dot_product = np.dot(unit_vector1, unit_vector2)

    # 내적을 바탕으로 두 단위벡터 사이의 각을 구한다.(라디안)
    angle_radians = math.acos(np.clip(dot_product, -1.0, 1.0))

    # 라디안 각을 degree로 변환
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

# 리스트에서 최대값의 인덱스를 찾는 함수
def find_max_index(lst):
    max_value = lst[0]
    max_index = 0

    for i in range(1, len(lst)):
        if lst[i] > max_value:
            max_value = lst[i]
            max_index = i

    return max_index





# 인공지능 뷰 (파이썬으로만 html 없이 opencv 화면 렌더링)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

class VideoCamera(object):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        success, image = self.cap.read()
        if not success:
            return None

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 오른쪽 허리 각도 계산 및 저장
            back_angle_right = round(
                calculateAngle3D(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]), 1)
            # 왼쪽 허리 각도 계산 및 저장
            back_angle_left = round(
                calculateAngle3D(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]), 1)
            
            # 오른쪽 무릎 각도 계산 및 저장
            knee_angle_right = round(
                calculateAngle3D(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]), 1)
            # 왼쪽 무릎 각도 계산 및 저장
            knee_angle_left = round(
                calculateAngle3D(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]), 1)
            
            # 발목-무릎-반대쪽 무릎 오른쪽 각도 계산 및 저장
            ankle_knee_knee_right = round(
                calculateAngle3D(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]), 1)
            # 발목-무릎-반대쪽 무릎 왼쪽 각도 계산 및 저장
            ankle_knee_knee_left = round(
                calculateAngle3D(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]), 1)
            
            # 무릎-엉덩이-반대쪽엉덩이 오른쪽 각도 계산 및 저장
            hip_hip_knee_right = round(
                calculateAngle3D(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]), 1)
            # 무릎-엉덩이-반대쪽엉덩이 왼쪽 각도 계산 및 저장
            hip_hip_knee_left = round(
                calculateAngle3D(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]), 1)
            
            input_data = np.array([[
                back_angle_right, back_angle_left,
                knee_angle_right, knee_angle_left,
                ankle_knee_knee_right, ankle_knee_knee_left,
                hip_hip_knee_right, hip_hip_knee_left
            ]])

            # 딥러닝 모델로 동작 분류
            # predictions = model.predict(input_data) # .h5
            predictions = model.predict_proba(input_data) # .joblib

            # 클래스 1 (올바른  동작)의 확률을 가져와 화면에 표시
            probability_class1 = predictions[0][1]  # 클래스 1에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)

            probability_class2 = predictions[0][2]  # 클래스 1에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)

            probability_class3 = predictions[0][3]  # 클래스 1에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)

            cv2.putText(image, f"Class 1 Probability: {probability_class1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.putText(image, f"Class 2 Probability: {probability_class2:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(image, f"Class 3 Probability: {probability_class3:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)




        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


