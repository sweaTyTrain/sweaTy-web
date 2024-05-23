# 라이브러리 설정
import modules
import cv2
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

video_route = './test/video/test3_3.mp4'
csv_save_route = './test/2D_point_csv/test3_3.csv'

# Initializing mediapipe pose class.
# mediapipe pose class를 초기화 한다.
mp_pose = mp.solutions.pose

# Initializing mediapipe drawing class, useful for annotation.
# mediapipe의 drawing class를 초기화한다.
mp_drawing = mp.solutions.drawing_utils

# pose detection function start
# 동영상 또는 웹캠 관절인식 결과 확인 코드
# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False,
                          min_tracking_confidence=0.1,
                          min_detection_confidence=0.8,
                          model_complexity=1,
                          smooth_landmarks=True)

# Initialize the VideoCapture object to read from the webcam.
# video = cv2.VideoCapture(0)

# Initialize the VideoCapture object to read from a video stored in the disk.
video = cv2.VideoCapture(video_route)

# 빈 데이터프레임 생성
df = pd.DataFrame(columns=['right_shoulder_x', 'right_shoulder_y',
                           'left_shoulder_x', 'left_shoulder_y',
                           'right_hip_x', 'right_hip_y',
                           'left_hip_x', 'left_hip_y',
                           'right_knee_x', 'right_knee_y',
                           'left_knee_x', 'left_knee_y',
                           'right_ankle_x', 'right_ankle_y',
                           'left_ankle_x', 'left_ankle_y'])

# 반복문 일시 정지를 위한 변수
paused = False

# csv 행을 확인하기 위한 변수
i = 1

# Iterate until the video is accessed successfully.
while video.isOpened():
    # Read a frame.
    hasFrame, frame = video.read()

    # Check if frame is not read properly.
    if not hasFrame:
        # Continue the loop.
        break

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the width and height of the frame
    frame_height, frame_width, _ = frame.shape

    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    # 관절 인식 수행
    results, frame, landmarks = modules.detectPose(frame, pose_video, mp_pose, mp_drawing, display=False)

    # 모든 landmark가 인식되었는지 확인
    if results.pose_world_landmarks is not None and all(results.pose_world_landmarks.landmark[j].visibility > 0.1 for j in [11, 12, 23, 24, 25, 26, 27, 28]):
        cv2.putText(frame, "all landmarks detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 여기서부터 점 좌표 정규화
        # 왼쪽 엉덩이 점을 (0, 0, 0)이 되도록 shift
        adjust_x = -1 * landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0]
        adjust_y = -1 * landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]
        #adjust_z = -1 * landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][2]

        landmarks_adjust_point = []

        for j in range(0, 33):
            landmarks_adjust_point.append((landmarks[j][0] + adjust_x,
                                           landmarks[j][1] + adjust_y,
                                           #landmarks[j][2] + adjust_z
                                           ))


        # # 왼쪽 엉덩이를 기준으로 정면 보도록 모든 좌표 회전
        # left_hip = np.array(landmarks_adjust_point[mp_pose.PoseLandmark.LEFT_HIP.value])
        # right_hip = np.array(landmarks_adjust_point[mp_pose.PoseLandmark.RIGHT_HIP.value])
        #
        # rotation_angle_y = math.degrees(math.atan2(right_hip[2] - left_hip[2], right_hip[0] - left_hip[0]))
        #
        # landmarks_rotated = rotate_around_left_hip(landmarks_adjust_point, 0, rotation_angle_y, 0)
        # landmarks_rotated = rotate_around_left_hip(landmarks_rotated, 270, 0, 180)

        # 엉덩이 사이의 거리를 1으로 하여 모든 관절을 정규화
        hip_distance = modules.calculateDistance2D(landmarks_adjust_point[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                   landmarks_adjust_point[mp_pose.PoseLandmark.RIGHT_HIP.value])

        landmarks_adjust_ratio = []

        for j in range(0, 33):
            normalized_x = landmarks_adjust_point[j][0] / hip_distance
            normalized_y = landmarks_adjust_point[j][1] / hip_distance

            landmarks_adjust_ratio.append((normalized_x, normalized_y))
        # 여기까지 점 좌표 정규화

        # # 좌표 추출
        # x_values = [point[0] for point in landmarks_adjust_ratio[:-1]]
        # y_values = [point[1] * -1 for point in landmarks_adjust_ratio[:-1]]
        #
        # # 플롯 생성
        # plt.scatter(x_values, y_values, marker='o')
        # plt.title('Scatter Plot of 32 Points')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.grid(True)
        # plt.show()

        # 데이터프레임에 새로운 행 추가
        now_points = [ round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_HIP.value][0], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_HIP.value][1], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_HIP.value][0], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_HIP.value][1], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_KNEE.value][0], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_KNEE.value][1], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_KNEE.value][0], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_KNEE.value][1], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_ANKLE.value][0], 2),
                       round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_ANKLE.value][1], 2)]
        print(now_points)
        df.loc[len(df)] = now_points
        i += 1
    # 관절이 하나도 인식되지 않았음 -> 오류 메세지 출력
    else:
        cv2.putText(frame, "some or all landmarks not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print("랜드마크 검출 불가")

    # 현재 프레임 번호 출력
    print(i, '번째 csv행')

    # 프레임 출력
    cv2.imshow('Pose Detection', frame)

    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed.
    if (k == 27):
        # Break the loop.
        break

    # 스페이스 바가 눌리면 paused값 not연산
    if k == 32:
        paused = not paused

    # 일시 정지 상태라면 아래 반복문을 무한 반복함
    if paused:
        while True:
            k = cv2.waitKey(0) & 0xFF
            # 다시 스페이스 바가 눌리면 paused값을 False로 바꾸고 반복분을 탈출함
            if k == 32:
                paused = False
                break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()

# csv파일로 저장
df.to_csv(csv_save_route, index=False)