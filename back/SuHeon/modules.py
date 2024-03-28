import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 관절 인식 함수
def detectPose(image, pose, mp_pose, mp_drawing, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    # 예시이미지 copy하기
    output_image = image.copy()

    # 컬러 이미지 BGR TO RGB 변환
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pose detection 수행
    results = pose.process(imageRGB)

    # input image의 너비&높이 탐색
    height, width, _ = image.shape

    # detection landmarks를 저장할 빈 list 초기화
    landmarks = []

    # landmark가 감지 되었는지 확인
    if results.pose_landmarks:

      # landmark 그리기
      mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

      # 감지된 landmark 반복
      for landmark in results.pose_landmarks.landmark:

        # landmark를 list에 추가하기
        # 이미지의 비율에 맞게 값을 곱해 정규화 해제
        # z값은 이미지의 비율을 알 수 없으므로 대충 너비의 나누기 3한 값을 곱해준다. -> 나중에 적당한 z가중치를 찾는 코드 추가해야 할 듯
        landmarks.append((float(landmark.x), float(landmark.y), float(landmark.z)))

    # 오리지널 image와 pose detect된 image 비교
    if display:
        # 3D 서브플롯을 생성합니다.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []

        for i in range(33):
            x.append(landmarks[i][0])
            y.append(landmarks[i][2])
            z.append(-1 * landmarks[i][1])

        # scatter 함수를 사용하여 3차원 점을 그립니다.
        ax.scatter(x, y, z)

        # 아래 옵션은 더 나은 시각화를 위해 조정되는 옵션임
        # 만일 모델이 찌그러져 나와도 실제 점은 landmarks에 제대로 찍혀있다.
        ax.set_box_aspect([30, 30, 100])  # sqat_img6.PNG 비율

        # 축 레이블을 설정합니다.
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 그래프를 표시합니다.
        plt.show()

        # # 오리지널 & 아웃풋 이미지 그리기
        # plt.figure(figsize=[17,17])
        #
        # plt.subplot(121)
        # plt.imshow(image[:,:,::-1])
        # plt.title("Original Image")
        # plt.axis('off')
        #
        # plt.subplot(122)
        # plt.imshow(output_image[:,:,::-1])
        # plt.title("Output Image")
        # plt.axis('off')

    return results, output_image, landmarks

# 2차원 각도계산 함수
def calculateAngle2D(landmark1, landmark2, landmark3):
    # 벡터를 만듭니다.
    vector1 = (landmark1[0] - landmark2[0], landmark1[1] - landmark2[1])
    vector2 = (landmark3[0] - landmark2[0], landmark3[1] - landmark2[1])

    # 벡터의 크기를 계산합니다.
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # 벡터의 내적을 계산합니다.
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # 각도를 계산합니다.
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_rad = math.acos(cos_theta)

    # 라디안을 각도로 변환하고 0~180 범위로 조정합니다.
    angle_deg = math.degrees(angle_rad) % 180

    return angle_deg

# 3차원 각도계산 함수
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

# 2차원 거리계산 함수
def calculateDistance2D(landmark1, landmark2):
    # point1와 point2는 각각 (x, y) 좌표를 담은 튜플 또는 리스트여야 합니다.
    x1, y1 = landmark1
    x2, y2 = landmark2

    # 각 좌표 축에서의 차이를 계산
    x_diff = x2 - x1
    y_diff = y2 - y1

    # 2차원 거리 계산
    distance = math.sqrt(x_diff**2 + y_diff**2)

    return distance

# 3차원 거리계산 함수
def calculateDistance3D(landmark1, landmark2):
    # point1와 point2는 각각 (x, y, z) 좌표를 담은 튜플 또는 리스트여야 합니다.
    x1, y1, z1 = landmark1
    x2, y2, z2 = landmark2

    # 각 좌표 축에서의 차이를 계산
    x_diff = x2 - x1
    y_diff = y2 - y1
    z_diff = z2 - z1

    # 3차원 거리 계산
    distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

    return distance

# 왼쪽 엉덩이 기준 정면 보게하는 함수
def rotate_around_left_hip(landmarks, mp_pose, rotation_angle_x=0, rotation_angle_y=0, rotation_angle_z=0):
    # 왼쪽 엉덩이의 좌표
    left_hip = np.array(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # 회전 행렬 생성
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, math.cos(math.radians(rotation_angle_x)), -math.sin(math.radians(rotation_angle_x))],
                                  [0, math.sin(math.radians(rotation_angle_x)), math.cos(math.radians(rotation_angle_x))]])

    rotation_matrix_y = np.array([[math.cos(math.radians(rotation_angle_y)), 0, math.sin(math.radians(rotation_angle_y))],
                                  [0, 1, 0],
                                  [-math.sin(math.radians(rotation_angle_y)), 0, math.cos(math.radians(rotation_angle_y))]])

    rotation_matrix_z = np.array([[math.cos(math.radians(rotation_angle_z)), -math.sin(math.radians(rotation_angle_z)), 0],
                                  [math.sin(math.radians(rotation_angle_z)), math.cos(math.radians(rotation_angle_z)), 0],
                                  [0, 0, 1]])

    # 회전 각도에 따라 각각 x, y, z 축으로 회전 적용
    rotated_landmarks = []
    for landmark in landmarks:
        rotated_landmark = np.array(landmark) - left_hip
        rotated_landmark = np.dot(rotation_matrix_x, rotated_landmark)
        rotated_landmark = np.dot(rotation_matrix_y, rotated_landmark)
        rotated_landmark = np.dot(rotation_matrix_z, rotated_landmark)
        rotated_landmark += left_hip
        rotated_landmarks.append(tuple(rotated_landmark))

    return rotated_landmarks

# 리스트에서 최대값의 인덱스를 찾는 함수
def find_max_index(lst):
    max_value = lst[0]
    max_index = 0

    for i in range(1, len(lst)):
        if lst[i] > max_value:
            max_value = lst[i]
            max_index = i

    return max_index