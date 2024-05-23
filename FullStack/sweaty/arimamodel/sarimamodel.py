#스쿼트 1회 앉았다 일어섰다

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import linregress
import math

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf





#거리 

def calculateDistance3D(landmark1, landmark2):
    
    # 각 좌표 축에서의 차이를 계산
    x_diff = landmark2.x-landmark1.x
    y_diff = landmark2.y-landmark1.y
    z_diff = landmark2.z-landmark1.z

    # 3차원 거리 계산
    distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

    return distance


#각도 

def calculateAngle3D(landmark1, landmark2, landmark3):
    # landmark1에서 landmark2로 향하는 3차원 벡터 계산
    vector1 = np.array([landmark1.x, landmark1.y, landmark1.z]) - np.array([landmark2.x, landmark2.y, landmark2.z])
    # landmark3에서 landmark2로 향하는 3차원 벡터 계산
    vector2 = np.array([landmark3.x, landmark3.y, landmark3.z]) - np.array([landmark2.x, landmark2.y, landmark2.z])

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









# 미디어 파이프 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 동영상 파일 경로
video_path = 'squat.mp4'







# 미디어 파이프 인스턴스 생성
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    # 양 무릎, 양 허리, 양 어깨의 좌표를 저장할 리스트
    keypoints_list = {'LEFT_KNEE': [], 'RIGHT_KNEE': [], 'LEFT_HIP': [], 'RIGHT_HIP': [], 'LEFT_SHOULDER': [], 'RIGHT_SHOULDER': []}
    
    
    
    #각도 거리를 저장할 리스트 
    
    keydistanceangle_list = { 'LEFT_KNEE_ANGLE':[], 'RIGHT_KNEE_ANGLE':[], 'LEFT_HIP_ANGLE':[],'RIGHT_HIP_ANGLE':[], 
                             'LEFT_INSIDE_KNEE_ANGLE':[],'RIGHT_INSIDE_KNEE_ANGLE':[],
                             'LHIP_TO_LFOOT':[], 'RHIP_TO_RFOOT':[], 'LHIP_TO_LKNEE':[],'RHIP_TO_RKNEE':[],'LSHOULDER_TO_LKNEE':[], 
                             'RSHOULDER_TO_RKNEE':[], 'LSHOULDER_TO_LFOOT':[],'RSHOULDER_TO_RFOOT':[],'LKNEE_TO_RKNEE':[],'LFOOT_TO_RFOOT':[]}
    
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
            
        #프레임 조정
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the width and height of the frame
        frame_height, frame_width, _ = frame.shape

        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        
        
        
        
        
        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 이미지에서 사람의 포즈 검출
        results = pose.process(image_rgb)
        
        # 검출 결과가 있는 경우
        if results.pose_landmarks:
            # 양 무릎, 양 허리, 양 어깨의 좌표를 저장
            landmarks = results.pose_landmarks.landmark
            keypoints_list['LEFT_KNEE'].append((landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z))
            keypoints_list['RIGHT_KNEE'].append((landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z))
            keypoints_list['LEFT_HIP'].append((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z))
            keypoints_list['RIGHT_HIP'].append((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z))
            keypoints_list['LEFT_SHOULDER'].append((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z))
            keypoints_list['RIGHT_SHOULDER'].append((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z))
            
            
            
            
            keydistanceangle_list['LEFT_KNEE_ANGLE'].append(calculateAngle3D(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                           
                                                                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))
            keydistanceangle_list['LEFT_HIP_ANGLE'].append(calculateAngle3D(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]))
            
            
            keydistanceangle_list['RIGHT_KNEE_ANGLE'].append(calculateAngle3D(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
            
            keydistanceangle_list['RIGHT_HIP_ANGLE'].append(calculateAngle3D(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]))
            
            
            keydistanceangle_list['LEFT_INSIDE_KNEE_ANGLE'].append(calculateAngle3D(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
            
            keydistanceangle_list['RIGHT_INSIDE_KNEE_ANGLE'].append(calculateAngle3D(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))
            
            
            
            
            
            
            keydistanceangle_list['LHIP_TO_LFOOT'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))
            
            keydistanceangle_list['LHIP_TO_LKNEE'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]))
            
            keydistanceangle_list['RHIP_TO_RFOOT'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
            
            keydistanceangle_list['RHIP_TO_RKNEE'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]))
            
            keydistanceangle_list['LSHOULDER_TO_LKNEE'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]))
            
            keydistanceangle_list['RSHOULDER_TO_RKNEE'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]))
            
            keydistanceangle_list['LSHOULDER_TO_LFOOT'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))
            
            keydistanceangle_list['RSHOULDER_TO_RFOOT'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
            
            
            keydistanceangle_list['LKNEE_TO_RKNEE'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]))
            
            
                        
            keydistanceangle_list['LFOOT_TO_RFOOT'].append(calculateDistance3D(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
            
            
            
            
            
        
    
        cv2.imshow('Pose Detection', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            
            
    # 동영상 분석이 끝나면 matplotlib를 사용하여 좌표를 3D 그래프와 2D 그래프로 표시
    fig = plt.figure(figsize=(12, 6))

    
    
    
    all_left_knee = keypoints_list['LEFT_KNEE']
    all_right_knee = keypoints_list['RIGHT_KNEE']
    all_left_hip = keypoints_list['LEFT_HIP']
    all_right_hip = keypoints_list['RIGHT_HIP']
    all_left_shoulder = keypoints_list['LEFT_SHOULDER']
    all_right_shoulder = keypoints_list['RIGHT_SHOULDER']
    
    x_left_knee, y_left_knee, z_left_knee = zip(*all_left_knee)
    x_right_knee, y_right_knee, z_right_knee = zip(*all_right_knee)
    x_left_hip, y_left_hip, z_left_hip = zip(*all_left_hip)
    x_right_hip, y_right_hip, z_right_hip = zip(*all_right_hip)
    x_left_shoulder, y_left_shoulder, z_left_shoulder = zip(*all_left_shoulder)
    x_right_shoulder, y_right_shoulder, z_right_shoulder = zip(*all_right_shoulder)
    
    
    
    
    
    
    
    #기울기 
    
    
    
    #print(x_left_knee)
    
    print('\n')
    #print( all_left_knee)
    
    
    #사이킷런 활용 부정확 
    
    #x,y,z 좌표 전체에 대해서 기울기 구하기 
    
    """
    # all_left_knee 좌표에서 x, y, z를 추출
    x_left_knee, y_left_knee, z_left_knee = zip(*all_left_knee)
    
    train_input,test_input,train_target,test_target = train_test_split(np.array([x_left_knee, y_left_knee, z_left_knee]).T, np.arange(len(all_left_knee)), random_state=42)

    model = LinearRegression()

    # x, y, z를 각각 독립 변수로, 좌표 인덱스를 종속 변수로 사용하여 모델 학습
    model.fit(train_input,train_target)

    # 기울기 계산
    slope_left_knee = model.coef_

    print("Left knee slope:", slope_left_knee)
    """
    
    #x좌표에 대해서 기울기 구하기 
    
    """
    # 데이터 준비
    x_data = np.array([keypoint[0] for keypoint in keypoints_list['LEFT_KNEE']]).reshape(-1, 1)
    y_data = np.arange(len(keypoints_list['LEFT_KNEE'])).reshape(-1, 1)

    # 훈련 및 테스트 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    # 선형 회귀 모델 생성 및 훈련
    model = LinearRegression()
    model.fit(x_train, y_train)

    # 기울기 출력
    print("기울기:", model.coef_[0][0])
    
    """
    
    
    #numpy 기울기 부정확 
        
    """
    #numpy 기울기 
    slope2, intercept = np.polyfit( x_left_knee,x_values, 1)   
    print("numpy - x_left_knee의 기울기:",slope2)
    
    """

    
    
    #scipy 라이브러리 활용
    
    # x_left_knee 데이터를 numpy 배열로 변환
    x_right_knee = np.array(x_left_knee)
    

    # x 좌표에 대한 데이터 포인트 수
    n_points = len(x_right_knee)
    
    
    # x 좌표에 대한 배열 생성 (1차원)
    x_values = np.arange(n_points)


    
    # SciPy의 linregress 함수를 사용하여 선형 회귀 수행
    slope, intercept, r_value, p_value, std_err = linregress(x_values, x_right_knee)
    # 기울기 출력
    print("scipy - x_left_knee의 기울기:", slope)
    
    
    
    

    
    
    

    
    
    





    # 3D 좌표 그래프  
    ax1 = fig.add_subplot(121, projection='3d')
    for joint, keypoints in keypoints_list.items():
        x, y, z = zip(*keypoints)
        ax1.plot(x, y, z, marker='o', linestyle='-', label=joint)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Pose Landmarks')
    ax1.legend()

    
    
    
    
    
    """
    # 2D 좌표 그래프
    ax2 = fig.add_subplot(122)
    for joint, keypoints in keypoints_list.items():
        x, y, z = zip(*keypoints)
        ax2.plot(x, y, marker='o', linestyle='-', label=joint)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('2D Pose Landmarks')
    ax2.legend()

    plt.show()
    """
    
    
    
    #  x,y,z 각 좌표에 대한 그래프 
    
    """
    # Hip
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(len(x_left_hip)), x_left_hip, marker='o', linestyle='-', color='r', label='X')
    plt.xlabel('Frame Index')
    plt.ylabel('X Coordinate')
    plt.title('X Coordinate of Left Hip Over Frames')
    plt.ylim(min(x_left_hip) - 0.5, max(x_left_hip) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(len(y_left_hip)), y_left_hip, marker='o', linestyle='-', color='g', label='Y')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Coordinate')
    plt.title('Y Coordinate of Left Hip Over Frames')
    plt.ylim(min(y_left_hip) - 0.5, max(y_left_hip) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(len(z_left_hip)), z_left_hip, marker='o', linestyle='-', color='b', label='Z')
    plt.xlabel('Frame Index')
    plt.ylabel('Z Coordinate')
    plt.title('Z Coordinate of Left Hip Over Frames')
    plt.ylim(min(z_left_hip) - 0.5, max(z_left_hip) + 0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(len(x_right_hip)), x_right_hip, marker='o', linestyle='-', color='r', label='X')
    plt.xlabel('Frame Index')
    plt.ylabel('X Coordinate')
    plt.title('X Coordinate of Right Hip Over Frames')
    plt.ylim(min(x_right_hip) - 0.5, max(x_right_hip) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(len(y_right_hip)), y_right_hip, marker='o', linestyle='-', color='g', label='Y')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Coordinate')
    plt.title('Y Coordinate of Right Hip Over Frames')
    plt.ylim(min(y_right_hip) - 0.5, max(y_right_hip) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(len(z_right_hip)), z_right_hip, marker='o', linestyle='-', color='b', label='Z')
    plt.xlabel('Frame Index')
    plt.ylabel('Z Coordinate')
    plt.title('Z Coordinate of Right Hip Over Frames')
    plt.ylim(min(z_right_hip) - 0.5, max(z_right_hip) + 0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # Knee
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(len(x_left_knee)), x_left_knee, marker='o', linestyle='-', color='r', label='X')
    plt.xlabel('Frame Index')
    plt.ylabel('X Coordinate')
    plt.title('X Coordinate of Left Knee Over Frames')
    plt.ylim(min(x_left_knee) - 0.5, max(x_left_knee) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(len(y_left_knee)), y_left_knee, marker='o', linestyle='-', color='g', label='Y')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Coordinate')
    plt.title('Y Coordinate of Left Knee Over Frames')
    plt.ylim(min(y_left_knee) - 0.5, max(y_left_knee) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(len(z_left_knee)), z_left_knee, marker='o', linestyle='-', color='b', label='Z')
    plt.xlabel('Frame Index')
    plt.ylabel('Z Coordinate')
    plt.title('Z Coordinate of Left Knee Over Frames')
    plt.ylim(min(z_left_knee) - 0.5, max(z_left_knee) + 0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(len(x_right_knee)), x_right_knee, marker='o', linestyle='-', color='r', label='X')
    plt.xlabel('Frame Index')
    plt.ylabel('X Coordinate')
    plt.title('X Coordinate of Right Knee Over Frames')
    plt.ylim(min(x_right_knee) - 0.5, max(x_right_knee) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(len(y_right_knee)), y_right_knee, marker='o', linestyle='-', color='g', label='Y')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Coordinate')
    plt.title('Y Coordinate of Right Knee Over Frames')
    plt.ylim(min(y_right_knee) - 0.5, max(y_right_knee) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(len(z_right_knee)), z_right_knee, marker='o', linestyle='-', color='b', label='Z')
    plt.xlabel('Frame Index')
    plt.ylabel('Z Coordinate')
    plt.title('Z Coordinate of Right Knee Over Frames')
    plt.ylim(min(z_right_knee) - 0.5, max(z_right_knee) + 0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # Shoulder
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(len(x_left_shoulder)), x_left_shoulder, marker='o', linestyle='-', color='r', label='X')
    plt.xlabel('Frame Index')
    plt.ylabel('X Coordinate')
    plt.title('X Coordinate of Left Shoulder Over Frames')
    plt.ylim(min(x_left_shoulder) - 0.5, max(x_left_shoulder) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(len(y_left_shoulder)), y_left_shoulder, marker='o', linestyle='-', color='g', label='Y')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Coordinate')
    plt.title('Y Coordinate of Left Shoulder Over Frames')
    plt.ylim(min(y_left_shoulder) - 0.5, max(y_left_shoulder) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(len(z_left_shoulder)), z_left_shoulder, marker='o', linestyle='-', color='b', label='Z')
    plt.xlabel('Frame Index')
    plt.ylabel('Z Coordinate')
    plt.title('Z Coordinate of Left Shoulder Over Frames')
    plt.ylim(min(z_left_shoulder) - 0.5, max(z_left_shoulder) + 0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(len(x_right_shoulder)), x_right_shoulder, marker='o', linestyle='-', color='r', label='X')
    plt.xlabel('Frame Index')
    plt.ylabel('X Coordinate')
    plt.title('X Coordinate of Right Shoulder Over Frames')
    plt.ylim(min(x_right_shoulder) - 0.5, max(x_right_shoulder) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(len(y_right_shoulder)), y_right_shoulder, marker='o', linestyle='-', color='g', label='Y')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Coordinate')
    plt.title('Y Coordinate of Right Shoulder Over Frames')
    plt.ylim(min(y_right_shoulder) - 0.5, max(y_right_shoulder) + 0.5)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(len(z_right_shoulder)), z_right_shoulder, marker='o', linestyle='-', color='b', label='Z')
    plt.xlabel('Frame Index')
    plt.ylabel('Z Coordinate')
    plt.title('Z Coordinate of Right Shoulder Over Frames')
    plt.ylim(min(z_right_shoulder) - 0.5, max(z_right_shoulder) + 0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    """
    
    
    
    
    
    #거리와 각도에 대한 그래프 
    


    # 각도와 거리 데이터 추출
    left_knee_angle = keydistanceangle_list['LEFT_KNEE_ANGLE']
    right_knee_angle = keydistanceangle_list['RIGHT_KNEE_ANGLE']
    left_hip_angle = keydistanceangle_list['LEFT_HIP_ANGLE']
    right_hip_angle = keydistanceangle_list['RIGHT_HIP_ANGLE']
    
    left_inside_knee_angle = keydistanceangle_list['LEFT_INSIDE_KNEE_ANGLE']
    right_inside_knee_angle = keydistanceangle_list['RIGHT_INSIDE_KNEE_ANGLE']
    
    
    
    lhip_to_lfoot = keydistanceangle_list['LHIP_TO_LFOOT']
    rhip_to_rfoot = keydistanceangle_list['RHIP_TO_RFOOT']
    lhip_to_lknee = keydistanceangle_list['LHIP_TO_LKNEE']
    rhip_to_rknee = keydistanceangle_list['RHIP_TO_RKNEE']
    lshoulder_to_lknee = keydistanceangle_list['LSHOULDER_TO_LKNEE']
    rshoulder_to_rknee = keydistanceangle_list['RSHOULDER_TO_RKNEE']
    lshoulder_to_lfoot = keydistanceangle_list['LSHOULDER_TO_LFOOT']
    rshoulder_to_rfoot = keydistanceangle_list['RSHOULDER_TO_RFOOT']
    lknee_to_rknee = keydistanceangle_list['LKNEE_TO_RKNEE']
    lfoot_to_rfoot = keydistanceangle_list['LFOOT_TO_RFOOT']
    
    
    
    
    
      
    # SARIMA 모델 학습  발 사이의 거리 예측 
    order = (1, 0, 0)  # ARIMA(p, d, q) 모델의 파라미터
    seasonal_order = (1, 0, 0, 170)  # SARIMA(P, D, Q, s) 모델의 파라미터   170은 프레임 주기 살짝 렉걸림 모델 분석 끝날때까지 기다리기 
    model = SARIMAX(lfoot_to_rfoot, order=order, seasonal_order=seasonal_order)
    result = model.fit()


    # 예측
    predictions = result.forecast(steps=170)  # 예측할 스텝 수를 조정해야 합니다.

    # 예측 결과 출력
    #print(predictions)



    # 시간에 따른 각도와 거리의 변화를 각각의 서브플롯에 표시
    plt.figure(figsize=(20, 12))

    # Left Knee Angle
    plt.subplot(4, 4, 1)
    plt.plot(left_knee_angle, label='Left Knee Angle', linestyle='-', marker='o', color='r')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle')
    plt.title('Left Knee Angle')
    plt.grid(True)

    # Right Knee Angle
    plt.subplot(4, 4, 2)
    plt.plot(right_knee_angle, label='Right Knee Angle', linestyle='-', marker='o', color='r')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle')
    plt.title('Right Knee Angle')
    plt.grid(True)

    # Left Hip Angle
    plt.subplot(4, 4, 3)
    plt.plot(left_hip_angle, label='Left Hip Angle', linestyle='-', marker='o', color='b')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle')
    plt.title('Left Hip Angle')
    plt.grid(True)

    # Right Hip Angle
    plt.subplot(4, 4, 4)
    plt.plot(right_hip_angle, label='Right Hip Angle', linestyle='-', marker='o', color='b')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle')
    plt.title('Right Hip Angle')
    plt.grid(True)
    
    
    # Left Inside Knee Angle
    plt.subplot(4, 4, 5)
    plt.plot(left_inside_knee_angle, label='Left Inside Knee Angle', linestyle='-', marker='o', color='g')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle')
    plt.title('Left Inside Knee Angle')
    plt.grid(True)
    
    # Right Inside Knee Angle
    plt.subplot(4, 4, 6)
    plt.plot(right_inside_knee_angle, label='Right Inside Knee Angle', linestyle='-', marker='o', color='g')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle')
    plt.title('Right Inside Knee Angle')
    plt.grid(True)
    

    # LHip to LFoot Distance
    plt.subplot(4, 4, 7)
    plt.plot(lhip_to_lfoot, label='LHip to LFoot Distance', linestyle='-', marker='o', color='m')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('LHip to LFoot Distance')
    plt.grid(True)

    # RHip to RFoot Distance
    plt.subplot(4, 4, 8)
    plt.plot(rhip_to_rfoot, label='RHip to RFoot Distance', linestyle='-', marker='o', color='m')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('RHip to RFoot Distance')
    plt.grid(True)

    # LHip to LKnee Distance
    plt.subplot(4, 4, 9)
    plt.plot(lhip_to_lknee, label='LHip to LKnee Distance', linestyle='-', marker='o', color='y')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('LHip to LKnee Distance')
    plt.grid(True)

    # RHip to RKnee Distance
    plt.subplot(4, 4, 10)
    plt.plot(rhip_to_rknee, label='RHip to RKnee Distance', linestyle='-', marker='o', color='y')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('RHip to RKnee Distance')
    plt.grid(True)

    # LShoulder to LKnee Distance
    plt.subplot(4, 4, 11)
    plt.plot(lshoulder_to_lknee, label='LShoulder to LKnee Distance', linestyle='-', marker='o', color='k')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('LShoulder to LKnee Distance')
    plt.grid(True)

    # RShoulder to RKnee Distance
    plt.subplot(4, 4, 12)
    plt.plot(rshoulder_to_rknee, label='RShoulder to RKnee Distance', linestyle='-', marker='o', color='k')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('RShoulder to RKnee Distance')
    plt.grid(True)

    # LShoulder to LFoot Distance
    plt.subplot(4, 4, 13)
    plt.plot(lshoulder_to_lfoot, label='LShoulder to LFoot Distance', linestyle='-', marker='o', color='tab:orange')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('LShoulder to LFoot Distance')
    plt.grid(True)

    # RShoulder to RFoot Distance
    plt.subplot(4, 4, 14)
    plt.plot(rshoulder_to_rfoot, label='RShoulder to RFoot Distance', linestyle='-', marker='o', color='tab:orange')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('RShoulder to RFoot Distance')
    plt.grid(True)
    
    
    
    """
 
    plt.subplot(4, 4, 15)
    plt.plot(lknee_to_rknee, label='LKnee to RKnee Distance', linestyle='-', marker='o', color='tab:brown')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('LKnee to RKnee Distance')
    plt.grid(True)
   """
    
    
    # 발 사이의 거리 
    plt.subplot(4, 4, 15)
    plt.plot(lfoot_to_rfoot, label='LFoot to RFoot Distance', linestyle='-', marker='o', color='tab:brown')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('LFoot to RFoot Distance')
    plt.grid(True)
    
    
    #sarima 모델을 활용한 다음 발 사이의 거리를 예측
    plt.subplot(4, 4, 16)
    plt.plot(predictions, label='LFoot to RFoot Distance', linestyle='-', marker='o', color='tab:brown')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance')
    plt.title('LFoot to RFoot Distance')
    plt.grid(True)
    




    


    plt.tight_layout()
    plt.show()


    

# 종료
cap.release()
cv2.destroyAllWindows()


