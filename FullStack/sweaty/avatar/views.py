from django.shortcuts import render
from django.http import StreamingHttpResponse
from .models import *
import numpy as np
import json
from django.http import JsonResponse
import logging


logger = logging.getLogger(__name__)




#홈 화면
def index(request):


    SampleDatas.objects.all().delete()
    csv_file_path = './avatar/AImodel/merged_file.csv'
    save_csv_to_database(csv_file_path)
    SquatDatatest.objects.all().delete()

    SquatDatatest.objects.create(
        squat_state = [0,0,0,0],
        accuracy = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        squat_count = 0,
        height = 0.0,
        half_height = 0.0,
        score = 0,
        reg = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    )


    # Keypoints 모델의 모든 객체 삭제
    Keypointstest.objects.all().delete()

    # KeyAngles 모델의 모든 객체 삭제
    Keyanglestest.objects.all().delete()
    # TestModel의 모든 레코드를 삭제
    TestModel.objects.all().delete()
    # TestModel 인스턴스 생성 및 squatCnt 속성을 0으로 설정
    test_model_instance = TestModel(squatCnt=0,
                                    squatBeforeState=1,
                                    squatNowState=1,
                                    squatState='',
                                    squatAccuracy=-1,
                                    stateQueue='-1,-1,-1,-1,-1')
    # 인스턴스를 저장하여 초기화된 데이터 생성
    test_model_instance.save()

    return render(request, 'index.html')



#인공지능 모델 뷰 (따로 html 파일을 만들 필요 X)
def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')



#아바타 뷰
def second(request):
    return render(request, 'second.html')


#firework 뷰
def firework(request):
    return render(request, 'firework.html')





"""
def test(request):
    if request.method == 'POST':
        rightShoulderXCoordinate = float(request.POST.get('rightShoulderXCoordinate'))
        rightShoulderYCoordinate = float(request.POST.get('rightShoulderYCoordinate'))
        rightShoulderZCoordinate = float(request.POST.get('rightShoulderZCoordinate'))
        leftShoulderXCoordinate = float(request.POST.get('leftShoulderXCoordinate'))
        leftShoulderYCoordinate = float(request.POST.get('leftShoulderYCoordinate'))
        leftShoulderZCoordinate = float(request.POST.get('leftShoulderZCoordinate'))

        rightHipXCoordinate = float(request.POST.get('rightHipXCoordinate'))
        rightHipYCoordinate = float(request.POST.get('rightHipYCoordinate'))
        rightHipZCoordinate = float(request.POST.get('rightHipZCoordinate'))
        leftHipXCoordinate = float(request.POST.get('leftHipXCoordinate'))
        leftHipYCoordinate = float(request.POST.get('leftHipYCoordinate'))
        leftHipZCoordinate = float(request.POST.get('leftHipZCoordinate'))

        rightKneeXCoordinate = float(request.POST.get('rightKneeXCoordinate'))
        rightKneeYCoordinate = float(request.POST.get('rightKneeYCoordinate'))
        rightKneeZCoordinate = float(request.POST.get('rightKneeZCoordinate'))
        leftKneeXCoordinate = float(request.POST.get('leftKneeXCoordinate'))
        leftKneeYCoordinate = float(request.POST.get('leftKneeYCoordinate'))
        leftKneeZCoordinate = float(request.POST.get('leftKneeZCoordinate'))

        rightAnkleXCoordinate = float(request.POST.get('rightAnkleXCoordinate'))
        rightAnkleYCoordinate = float(request.POST.get('rightAnkleYCoordinate'))
        rightAnkleZCoordinate = float(request.POST.get('rightAnkleZCoordinate'))
        leftAnkleXCoordinate = float(request.POST.get('leftAnkleXCoordinate'))
        leftAnkleYCoordinate = float(request.POST.get('leftAnkleYCoordinate'))
        leftAnkleZCoordinate = float(request.POST.get('leftAnkleZCoordinate'))
        landmarkData = request.POST.get('landmarkData')


        # 오른쪽 허리 각도 계산 및 저장
        back_angle_right = round(
            calculateAngle3D_2(rightShoulderXCoordinate, rightShoulderYCoordinate, rightShoulderZCoordinate,
                               rightHipXCoordinate, rightHipYCoordinate, rightHipZCoordinate,
                               rightKneeXCoordinate, rightKneeYCoordinate, rightKneeZCoordinate), 1)
        # 왼쪽 허리 각도 계산 및 저장
        back_angle_left = round(
            calculateAngle3D_2(leftShoulderXCoordinate, leftShoulderYCoordinate, leftShoulderZCoordinate,
                               leftHipXCoordinate, leftHipYCoordinate, leftHipZCoordinate,
                               leftKneeXCoordinate, leftKneeYCoordinate, leftKneeZCoordinate), 1)

        # 오른쪽 무릎 각도 계산 및 저장
        knee_angle_right = round(
            calculateAngle3D_2(rightHipXCoordinate, rightHipYCoordinate, rightHipZCoordinate,
                               rightKneeXCoordinate, rightKneeYCoordinate, rightKneeZCoordinate,
                               rightAnkleXCoordinate, rightAnkleYCoordinate, rightAnkleZCoordinate), 1)
        # 왼쪽 무릎 각도 계산 및 저장
        knee_angle_left = round(
            calculateAngle3D_2(leftHipXCoordinate, leftHipYCoordinate, leftHipZCoordinate,
                               leftKneeXCoordinate, leftKneeYCoordinate, leftKneeZCoordinate,
                               leftAnkleXCoordinate, leftAnkleYCoordinate, leftAnkleZCoordinate), 1)

        # 발목-무릎-반대쪽 무릎 오른쪽 각도 계산 및 저장
        ankle_knee_knee_right = round(
            calculateAngle3D_2(rightAnkleXCoordinate, rightAnkleYCoordinate, rightAnkleZCoordinate,
                               rightKneeXCoordinate, rightKneeYCoordinate, rightKneeZCoordinate,
                               leftKneeXCoordinate, leftKneeYCoordinate, leftKneeZCoordinate), 1)
        # 발목-무릎-반대쪽 무릎 왼쪽 각도 계산 및 저장
        ankle_knee_knee_left = round(
            calculateAngle3D_2(leftAnkleXCoordinate, leftAnkleYCoordinate, leftAnkleZCoordinate,
                               leftKneeXCoordinate, leftKneeYCoordinate, leftKneeZCoordinate,
                               rightKneeXCoordinate, rightKneeYCoordinate, rightKneeZCoordinate), 1)

        # 무릎-엉덩이-반대쪽엉덩이 오른쪽 각도 계산 및 저장
        hip_hip_knee_right = round(
            calculateAngle3D_2(rightKneeXCoordinate, rightKneeYCoordinate, rightKneeZCoordinate,
                               rightHipXCoordinate, rightHipYCoordinate, rightHipZCoordinate,
                               leftHipXCoordinate, leftHipYCoordinate, leftHipZCoordinate), 1)
        # 무릎-엉덩이-반대쪽엉덩이 왼쪽 각도 계산 및 저장
        hip_hip_knee_left = round(
            calculateAngle3D_2(leftKneeXCoordinate, leftKneeYCoordinate, leftKneeZCoordinate,
                               leftHipXCoordinate, leftHipYCoordinate, leftHipZCoordinate,
                               rightHipXCoordinate, rightHipYCoordinate, rightHipZCoordinate), 1)


        input_data = np.array([[
            back_angle_right, back_angle_left,
            knee_angle_right, knee_angle_left,
            ankle_knee_knee_right, ankle_knee_knee_left,
            hip_hip_knee_right, hip_hip_knee_left
        ]])

        #print(input_data)

        # 딥러닝 모델로 동작 분류
        # predictionsTest = model.predict(input_data) # .h5
        predictionsTest = model.predict_proba(input_data) # .joblib

        # 클래스 1 (올바른  동작)의 확률을 가져와 화면에 표시
        Nprobability_class0 = round(predictionsTest[0][0], 5)  # 클래스 0에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)
        Nprobability_class1 = round(predictionsTest[0][1], 5)  # 클래스 1에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)
        Nprobability_class2 = round(predictionsTest[0][2], 5)  # 클래스 2에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)
        Nprobability_class3 = round(predictionsTest[0][3], 5)  # 클래스 3에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)
        Nprobability_class4 = round(predictionsTest[0][4], 5)  # 클래스 4에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)
        Nprobability_class5 = round(predictionsTest[0][5], 5)  # 클래스 5에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)
        Nprobability_class6 = round(predictionsTest[0][6], 5)  # 클래스 6에 해당하는 확률 (0은 클래스 0, 1은 클래스 1)

        Cprobability_class0 = str(Nprobability_class0)
        Cprobability_class1 = str(Nprobability_class1)
        Cprobability_class2 = str(Nprobability_class2)
        Cprobability_class3 = str(Nprobability_class3)
        Cprobability_class4 = str(Nprobability_class4)
        Cprobability_class5 = str(Nprobability_class5)
        Cprobability_class6 = str(Nprobability_class6)

        latest_entry = TestModel.objects.latest('id')
        # 스쿼트 카운트 관련 변수
        squatCnt = latest_entry.squatCnt
        squatBeforeState = latest_entry.squatBeforeState
        squatNowState = latest_entry.squatNowState
        squatAccuracy = latest_entry.squatAccuracy
        classIdx = latest_entry.classIdx

        pre = [Nprobability_class0, Nprobability_class1, Nprobability_class2,
               Nprobability_class3, Nprobability_class4, Nprobability_class5,
               Nprobability_class6]

        classIdx = int(classIdx)
        classIdx_temp = find_max_index(pre)
        stateQueue = [int(i) for i in latest_entry.stateQueue.split(',')]

        if len(latest_entry.squatState) == 0:
            squatState = []
        elif len(latest_entry.squatState) == 1:
            squatState = [int(latest_entry.squatState)]
        else:
            squatState = [int(i) for i in latest_entry.squatState.split(',')]

        # 연속 5프레임 똑같은 class가 나오면 제대로 인식했다고 판정
        stateQueue.pop(0)
        stateQueue.append(classIdx_temp)

        # 모두 같은지 판별
        all_same = all(element == stateQueue[0] for element in stateQueue)

        # 현재 클래스 보정
        if all_same:
            classIdx = stateQueue[0]

        if classIdx > 0 and classIdx <= 2:
            squatState.append(1)
        elif classIdx > 2 and classIdx <= 6:
            squatState.append(0)

        # 예측클래스가 0 이라면 일어서 있는것으로 판단
        if classIdx == 0:
            squatNowState = '1'
            latest_entry.squatNowState = '1'
        # 예측클래스가 2,4,6 이라면 앉아 있는것으로 판단
        elif classIdx == 2 or classIdx == 4 or classIdx == 6:
            squatNowState = '0'
            latest_entry.squatNowState = '0'
        latest_entry.save()

        latest_entry = TestModel.objects.latest('id')
        squatBeforeState = latest_entry.squatBeforeState
        squatNowState = latest_entry.squatNowState

        # 만약 squatBeforeState(이전 상태)가 0(앉은 상태)였는데
        # squatNowState(현재 상태)가 1(서있는 상태)가 되면 squatCnt증가
        if squatBeforeState == '0' and squatNowState == '1':
            squatAccuracy = round(sum(squatState) / len(squatState), 5)

            if squatAccuracy > 0.7:
                squatCnt = str(int(squatCnt) + 1)

            squatState = []

        # 다음 프레임을 받아오기 전에 squatNowState를 squatBeforeState에 기억시킨다.
        squatBeforeState = squatNowState

        squatAccuracy = str(squatAccuracy)

        stateQueue = ','.join(map(str, stateQueue))
        if len(squatState) == 0:
            squatState = ''
        elif len(squatState) == 1:
            squatState = str(squatState[0])
        else:
            squatState = ','.join(map(str, squatState))

        classIdx = str(classIdx)

        print('squatAccuracy: ', squatAccuracy)
        print('squatBeforeState: ', squatBeforeState)
        print('squatCnt: ', squatCnt)
        print('squatNowState: ', squatNowState)
        print('squatState: ', squatState)
        print('stateQueue: ', stateQueue)
        print('classIdx: ', classIdx)
        # DB에 저장
        TestModel.objects.create(label_0=Cprobability_class0, label_1=Cprobability_class1, label_2=Cprobability_class2,
                                 label_3=Cprobability_class3, label_4=Cprobability_class4, label_5=Cprobability_class5,
                                 label_6=Cprobability_class6, squatAccuracy=squatAccuracy, squatBeforeState=squatBeforeState,
                                 squatCnt=squatCnt, squatNowState=squatNowState, squatState=squatState, stateQueue=stateQueue,
                                 classIdx=classIdx)

    latest_entry = TestModel.objects.latest('id')
    #TestModel.objects.filter(id__lt=latest_entry.id).delete()
    json_data0 = latest_entry.label_0
    json_data1 = latest_entry.label_1
    json_data2 = latest_entry.label_2
    json_data3 = latest_entry.label_3
    json_data4 = latest_entry.label_4
    json_data5 = latest_entry.label_5
    json_data6 = latest_entry.label_6
    json_data7 = latest_entry.squatAccuracy
    json_data8 = latest_entry.squatBeforeState
    json_data9 = latest_entry.squatCnt
    json_data10 = latest_entry.squatNowState
    json_data11 = latest_entry.squatState
    json_data12 = latest_entry.stateQueue
    json_data13 = latest_entry.classIdx


    return JsonResponse({'json_data0': json_data0, 'json_data1': json_data1, 'json_data2': json_data2,
                         'json_data3': json_data3, 'json_data4': json_data4, 'json_data5': json_data5,
                         'json_data6': json_data6, 'json_data7': json_data7, 'json_data8': json_data8,
                         'json_data9': json_data9, 'json_data10': json_data10, 'json_data11': json_data11,
                         'json_data12': json_data12, 'json_data13': json_data13})


def is_json(data):
    try:
        json.loads(data)
        return True
    except ValueError:
        return False

"""


def test(request):
    if request.method == 'POST':
        rightShoulderXCoordinate = float(request.POST.get('rightShoulderXCoordinate'))
        rightShoulderYCoordinate = float(request.POST.get('rightShoulderYCoordinate'))
        #rightShoulderZCoordinate = float(request.POST.get('rightShoulderZCoordinate'))
        leftShoulderXCoordinate = float(request.POST.get('leftShoulderXCoordinate'))
        leftShoulderYCoordinate = float(request.POST.get('leftShoulderYCoordinate'))
        #leftShoulderZCoordinate = float(request.POST.get('leftShoulderZCoordinate'))


        rightHipXCoordinate = float(request.POST.get('rightHipXCoordinate'))
        rightHipYCoordinate = float(request.POST.get('rightHipYCoordinate'))
        #rightHipZCoordinate = float(request.POST.get('rightHipZCoordinate'))
        leftHipXCoordinate = float(request.POST.get('leftHipXCoordinate'))
        leftHipYCoordinate = float(request.POST.get('leftHipYCoordinate'))
        #leftHipZCoordinate = float(request.POST.get('leftHipZCoordinate'))

        rightKneeXCoordinate = float(request.POST.get('rightKneeXCoordinate'))
        rightKneeYCoordinate = float(request.POST.get('rightKneeYCoordinate'))
        #rightKneeZCoordinate = float(request.POST.get('rightKneeZCoordinate'))
        leftKneeXCoordinate = float(request.POST.get('leftKneeXCoordinate'))
        leftKneeYCoordinate = float(request.POST.get('leftKneeYCoordinate'))
        #leftKneeZCoordinate = float(request.POST.get('leftKneeZCoordinate'))

        rightAnkleXCoordinate = float(request.POST.get('rightAnkleXCoordinate'))
        rightAnkleYCoordinate = float(request.POST.get('rightAnkleYCoordinate'))
        #rightAnkleZCoordinate = float(request.POST.get('rightAnkleZCoordinate'))
        leftAnkleXCoordinate = float(request.POST.get('leftAnkleXCoordinate'))
        leftAnkleYCoordinate = float(request.POST.get('leftAnkleYCoordinate'))
        #leftAnkleZCoordinate = float(request.POST.get('leftAnkleZCoordinate'))

        rightHeelXCoordinate = float(request.POST.get('rightHeelXCoordinate'))
        rightHeelYCoordinate = float(request.POST.get('rightHeelYCoordinate'))
        #rightHeelZCoordinate = float(request.POST.get('rightHeelZCoordinate'))
        leftHeelXCoordinate = float(request.POST.get('leftHeelXCoordinate'))
        leftHeelYCoordinate = float(request.POST.get('leftHeelYCoordinate'))
        #leftHeelZCoordinate = float(request.POST.get('leftHeelZCoordinate'))

        rightFootXCoordinate = float(request.POST.get('rightFootXCoordinate'))
        rightFootYCoordinate = float(request.POST.get('rightFootYCoordinate'))
        #rightHeelZCoordinate = float(request.POST.get('rightHeelZCoordinate'))
        leftFootXCoordinate = float(request.POST.get('leftFootXCoordinate'))
        leftFootYCoordinate = float(request.POST.get('leftFootYCoordinate'))
        #leftHeelZCoordinate = float(request.POST.get('leftHeelZCoordinate'))



         # 정확도 자세를 위한 점수 배열 합산해서 판단

        # 0번에서 5번 인덱스  무릎, 힙, 어깨 인덱스 움직임 점수, 움직임이 적어야 점수를 줌

        # 6번에서 8번은 내려갈 때 무릎,힙, 어깨 기울기 9에서 10번은 내려갈 때 허리각도

        # 11번에서 13번은 올라갈 때 무릎,힙,어깨 기울기 14에서 15번은 올라갈 때 허리각도

        # 16번에서 18번은 양 발 사이의 거리 점수

        # 0,0,0,0 일어선 상태 150도 이상
        # 1,0,0,0 무릎이 150도 이하 100도이상
        # 1,1,0,0 무릎이 100도 이하
        # 1,1,1,0 무릎이 100도 이상 150도이하
        # 1,1,1,1 무릎이 150도 이상




        left_knee_angle = calculateAngle2D(leftHipXCoordinate, leftKneeXCoordinate, leftAnkleXCoordinate, leftHipYCoordinate, leftKneeYCoordinate, leftAnkleYCoordinate)
        right_knee_angle = calculateAngle2D(rightHipXCoordinate,rightKneeXCoordinate,rightAnkleXCoordinate, rightHipYCoordinate,rightKneeYCoordinate,rightAnkleYCoordinate)
        left_hip_angle = calculateAngle2D(leftShoulderXCoordinate, leftHipXCoordinate, leftKneeXCoordinate, leftShoulderYCoordinate,leftHipYCoordinate, leftKneeYCoordinate)
        right_hip_angle = calculateAngle2D(rightShoulderXCoordinate, rightHipXCoordinate, rightKneeXCoordinate, rightShoulderYCoordinate, rightHipYCoordinate, rightKneeYCoordinate)
        left_inside_hip_angle = calculateAngle2D(leftKneeXCoordinate, leftHipXCoordinate, rightHipXCoordinate, leftKneeYCoordinate, leftHipYCoordinate, rightHipYCoordinate)
        right_inside_hip_angle = calculateAngle2D(rightKneeXCoordinate, rightHipXCoordinate, leftHipXCoordinate, rightKneeYCoordinate, rightHipYCoordinate, leftHipYCoordinate)

        # 어깨 대비 발 너비 비율 구하기

        ratio1 = substract_x(leftAnkleXCoordinate, rightAnkleXCoordinate) / substract_x(leftShoulderXCoordinate,
                                                                                        rightShoulderXCoordinate)

        ratio2 = substract_x(leftHeelXCoordinate, rightHeelXCoordinate) / substract_x(leftShoulderXCoordinate,
                                                                                      rightShoulderXCoordinate)

        ratio3 = substract_x(leftFootXCoordinate, rightFootXCoordinate) / substract_x(leftShoulderXCoordinate,
                                                                                      rightShoulderXCoordinate)

        # 1. 무릎각도가 150도 이상일때 0,0,0,0
        # 2. 150도 이하 엉덩이>무릎 일때 1,0,0,0
        # 3. 엉덩이<무릎 일때 1,1,0,0
        # 4. 엉덩이>무릎 150도 이하 일때 1,1,1,0
        # 5. 무릎각도가 150도 이상일때 1,1,1,1



        foot_to_shoulder = substract_y(leftAnkleYCoordinate
                                ,leftShoulderYCoordinate
                                ,rightAnkleYCoordinate
                                ,rightShoulderYCoordinate)

        foot_to_hip = substract_y(leftAnkleYCoordinate
                                ,leftHipYCoordinate
                                ,rightAnkleYCoordinate
                                ,rightHipYCoordinate)


        squat_data = SquatDatatest.objects.latest('id')


        if squat_data.squat_state == [0,0,0,0] and (left_knee_angle + right_knee_angle) / 2 >= 150:
            height = substract_y(leftAnkleYCoordinate,leftShoulderYCoordinate,rightAnkleYCoordinate,rightShoulderYCoordinate)
            half_height = substract_y(leftAnkleYCoordinate,leftHipYCoordinate,rightAnkleYCoordinate,rightHipYCoordinate)


            squat_data.height = height
            squat_data.half_height = half_height

            squat_data.accuracy = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

            squat_data.save()



        if  (left_knee_angle + right_knee_angle) / 2 < 150 and (left_knee_angle + right_knee_angle) / 2 > 120:
            Keypointstest.objects.create(left_kneex=leftKneeXCoordinate,
                                         right_kneex=rightKneeXCoordinate,
                                         left_hipx=leftHipXCoordinate,
                                         right_hipx=rightHipXCoordinate,
                                         left_shoulderx=leftShoulderXCoordinate,
                                         right_shoulderx=rightShoulderXCoordinate,
                                         left_kneey=leftKneeYCoordinate,
                                         right_kneey=rightKneeYCoordinate,
                                         left_hipy=leftHipYCoordinate,
                                         right_hipy=rightHipYCoordinate,
                                         left_shouldery=leftShoulderYCoordinate,
                                         right_shouldery=rightShoulderYCoordinate,
                                         left_ankle=leftAnkleXCoordinate,
                                         right_ankle=rightAnkleXCoordinate,
                                         left_heel=leftHeelXCoordinate,
                                         right_heel=rightHeelXCoordinate,
                                         left_foot=leftFootXCoordinate,
                                         right_foot=rightFootXCoordinate)
            Keyanglestest.objects.create(left_knee_angle=left_knee_angle,
                                         right_knee_angle=right_knee_angle,
                                         left_hip_angle=left_hip_angle,
                                         right_hip_angle=right_hip_angle,
                                         left_inside_hip_angle=left_inside_hip_angle,
                                         right_inside_hip_angle=right_inside_hip_angle,
                                         ratio_foot_to_shoulder=foot_to_shoulder/squat_data.height,
                                         ratio_foot_to_hip=foot_to_hip/squat_data.half_height)


            x_left_knee = Keypointstest.objects.values_list('left_kneex', flat=True)
            x_right_knee = Keypointstest.objects.values_list('right_kneex', flat=True)
            x_left_hip = Keypointstest.objects.values_list('left_hipx', flat=True)
            x_right_hip = Keypointstest.objects.values_list('right_hipx', flat=True)
            x_left_shoulder = Keypointstest.objects.values_list('left_shoulderx', flat=True)
            x_right_shoulder = Keypointstest.objects.values_list('right_shoulderx', flat=True)

            y_left_knee = Keypointstest.objects.values_list('left_kneey', flat=True)
            y_right_knee = Keypointstest.objects.values_list('right_kneey', flat=True)
            y_left_hip = Keypointstest.objects.values_list('left_hipy', flat=True)
            y_right_hip = Keypointstest.objects.values_list('right_hipy', flat=True)
            y_left_shoulder = Keypointstest.objects.values_list('left_shouldery', flat=True)
            y_right_shoulder = Keypointstest.objects.values_list('right_shouldery', flat=True)




            left_knee_angles = Keyanglestest.objects.values_list('left_knee_angle', flat=True)
            # 오른쪽 무릎 각도 데이터 추출
            right_knee_angles = Keyanglestest.objects.values_list('right_knee_angle', flat=True)
            # 왼쪽 엉덩이 각도 데이터 추출
            left_hip_angles = Keyanglestest.objects.values_list('left_hip_angle', flat=True)
            # 오른쪽 엉덩이 각도 데이터 추출
            right_hip_angles = Keyanglestest.objects.values_list('right_hip_angle', flat=True)
            # 왼쪽 내부 엉덩이 각도 데이터 추출
            left_inside_hip_angles = Keyanglestest.objects.values_list('left_inside_hip_angle', flat=True)
            # 오른쪽 내부 엉덩이 각도 데이터 추출
            right_inside_hip_angles = Keyanglestest.objects.values_list('right_inside_hip_angle', flat=True)

            ratio_foot_to_shoulder = Keyanglestest.objects.values_list('ratio_foot_to_shoulder', flat=True)

            ratio_foot_to_hip = Keyanglestest.objects.values_list('ratio_foot_to_hip', flat=True)



            reg_knee_coef = calculate_coef(left_knee_angles,right_knee_angles,ratio_foot_to_hip).coef_

            reg_hip_coef = calculate_coef(left_hip_angles,right_hip_angles,ratio_foot_to_hip).coef_

            reg_inside_hip_coef = calculate_coef(left_inside_hip_angles,right_inside_hip_angles, ratio_foot_to_hip).coef_

            waist_coef = calculate_coef(left_knee_angles,right_knee_angles,ratio_foot_to_shoulder).coef_


            reg_left_shoulder =coord_coef( x_left_shoulder, y_left_shoulder).coef_
            reg_right_shoulder = coord_coef(x_right_shoulder, y_right_shoulder).coef_
            reg_left_hip = coord_coef(x_left_hip, y_left_hip).coef_
            reg_right_hip = coord_coef(x_right_hip, y_right_hip).coef_
            reg_left_knee = coord_coef(x_left_knee, y_left_knee).coef_
            reg_right_knee = coord_coef(x_right_knee, y_right_knee).coef_



            squat_data.reg[0] = float(reg_left_shoulder)
            squat_data.reg[1] = float(reg_right_shoulder)
            squat_data.reg[2] = float(reg_left_hip)
            squat_data.reg[3] = float(reg_right_hip)
            squat_data.reg[4] = float(reg_left_knee)
            squat_data.reg[5] = float(reg_right_knee)
            squat_data.reg[6] = float(reg_knee_coef)
            squat_data.reg[7] = float(reg_hip_coef)
            squat_data.reg[8] = float(reg_inside_hip_coef)
            squat_data.reg[9] = float(waist_coef)
            squat_data.reg[10] = float(reg_left_shoulder)
            squat_data.reg[11] = float(reg_right_shoulder)
            squat_data.reg[12] = float(reg_left_hip)
            squat_data.reg[13] = float(reg_right_hip)
            squat_data.reg[14] = float(reg_left_knee)
            squat_data.reg[15] = float(reg_right_knee)
            squat_data.reg[16] = float(reg_knee_coef)
            squat_data.reg[17] = float(reg_hip_coef)
            squat_data.reg[18] = float(reg_inside_hip_coef)
            squat_data.reg[19] = float(waist_coef)




            squat_data.save()

            if squat_data.squat_state==[0,0,0,0]:
                squat_data.squat_state = [1,0,0,0]
                squat_data.save()

            if squat_data.squat_state==[1,1,0,0]:
                squat_data.squat_state = [1,1,1,0]
                squat_data.save()




        if squat_data.squat_state == [1,0,0,0] and (left_knee_angle + right_knee_angle) / 2 <= 120:

            if (model[list(model.keys())[3]].score_samples(np.array(squat_data.reg[0]).reshape(-1, 1))<90):
                squat_data.accuracy[0] = 1
                squat_data.save()

            if (model[list(model.keys())[4]].score_samples(np.array(squat_data.reg[1]).reshape(-1, 1))<90):
                squat_data.accuracy[1] = 1
                squat_data.save()

            if (model[list(model.keys())[5]].score_samples(np.array(squat_data.reg[2]).reshape(-1, 1))<90):
                squat_data.accuracy[2] = 1
                squat_data.save()

            if (model[list(model.keys())[6]].score_samples(np.array(squat_data.reg[3]).reshape(-1, 1)) < 90):
                squat_data.accuracy[3] = 1
                squat_data.save()

            if (model[list(model.keys())[7]].score_samples(np.array(squat_data.reg[4]).reshape(-1, 1)) < 90):
                squat_data.accuracy[4] = 1
                squat_data.save()

            if (model[list(model.keys())[8]].score_samples(np.array(squat_data.reg[5]).reshape(-1, 1)) < 90):
                squat_data.accuracy[5] = 1
                squat_data.save()

            if (model[list(model.keys())[9]].score_samples(np.array(squat_data.reg[6]).reshape(-1, 1))<90):
                squat_data.accuracy[6] = 1
                squat_data.save()

            if (model[list(model.keys())[10]].score_samples(np.array(squat_data.reg[7]).reshape(-1, 1))<90):
                squat_data.accuracy[7] = 1
                squat_data.save()

            if (model[list(model.keys())[11]].score_samples(np.array(squat_data.reg[8]).reshape(-1, 1))<90):
                squat_data.accuracy[8] = 1
                squat_data.save()

            if (model[list(model.keys())[12]].score_samples(np.array(squat_data.reg[9]).reshape(-1, 1))<90):
                squat_data.accuracy[9] = 1
                squat_data.save()



            # 발 너비 점수
            if (model[list(model.keys())[0]].score_samples(np.array(ratio1).reshape(-1, 1)) < 90):
                squat_data.accuracy[20] = 1
                squat_data.save()

            if (model[list(model.keys())[1]].score_samples(np.array(ratio2).reshape(-1, 1)) < 90):
                squat_data.accuracy[21] = 1
                squat_data.save()

            if (model[list(model.keys())[2]].score_samples(np.array(ratio3).reshape(-1, 1)) < 90):
                squat_data.accuracy[22] = 1
                squat_data.save()


            # Keypoints 모델의 모든 객체 삭제
            Keypointstest.objects.all().delete()
            # KeyAngles 모델의 모든 객체 삭제
            Keyanglestest.objects.all().delete()

            squat_data.squat_state = [1,1,0,0]
            squat_data.save()







        if squat_data.squat_state==[1,1,1,0] and (left_knee_angle + right_knee_angle) / 2 >= 150:

            if (model[list(model.keys())[13]].score_samples(np.array(squat_data.reg[10]).reshape(-1, 1)) < 90):
                squat_data.accuracy[10] = 1
                squat_data.save()

            if (model[list(model.keys())[14]].score_samples(np.array(squat_data.reg[11]).reshape(-1, 1)) < 90):
                squat_data.accuracy[11] = 1
                squat_data.save()

            if (model[list(model.keys())[15]].score_samples(np.array(squat_data.reg[12]).reshape(-1, 1)) < 90):
                squat_data.accuracy[12] = 1
                squat_data.save()

            if (model[list(model.keys())[16]].score_samples(np.array(squat_data.reg[13]).reshape(-1, 1)) < 90):
                squat_data.accuracy[13] = 1
                squat_data.save()

            if (model[list(model.keys())[17]].score_samples(np.array(squat_data.reg[14]).reshape(-1, 1)) < 90):
                squat_data.accuracy[14] = 1
                squat_data.save()

            if (model[list(model.keys())[18]].score_samples(np.array(squat_data.reg[15]).reshape(-1, 1)) < 90):
                squat_data.accuracy[15] = 1
                squat_data.save()

            if (model[list(model.keys())[19]].score_samples(np.array(squat_data.reg[16]).reshape(-1, 1)) < 90):
                squat_data.accuracy[16] = 1
                squat_data.save()

            if (model[list(model.keys())[20]].score_samples(np.array(squat_data.reg[17]).reshape(-1, 1)) < 90):
                squat_data.accuracy[17] = 1
                squat_data.save()

            if (model[list(model.keys())[21]].score_samples(np.array(squat_data.reg[18]).reshape(-1, 1)) < 90):
                squat_data.accuracy[18] = 1
                squat_data.save()

            if (model[list(model.keys())[22]].score_samples(np.array(squat_data.reg[19]).reshape(-1, 1)) < 90):
                squat_data.accuracy[19] = 1
                squat_data.save()


                # 발 너비 점수
            if (model[list(model.keys())[0]].score_samples(np.array(ratio1).reshape(-1, 1)) < 90):
                squat_data.accuracy[20] = 1
                squat_data.save()

            if (model[list(model.keys())[1]].score_samples(np.array(ratio2).reshape(-1, 1)) < 90):
                squat_data.accuracy[21] = 1
                squat_data.save()

            if (model[list(model.keys())[2]].score_samples(np.array(ratio3).reshape(-1, 1)) < 90):
                squat_data.accuracy[22] = 1
                squat_data.save()


            squat_data.squat_state = [1,1,1,1]
            squat_data.save()


        #스코어 계산
        if squat_data.squat_state == [1,1,1,1]:
            # Keypoints 모델의 모든 객체 삭제
            Keypointstest.objects.all().delete()
            # KeyAngles 모델의 모든 객체 삭제
            Keyanglestest.objects.all().delete()
            squat_data.score = 100 - 4 * sum(squat_data.accuracy)
            if (squat_data.score>=80):
                squat_data.squat_count = squat_data.squat_count+1

            squat_data.squat_state=[0,0,0,0]
            squat_data.save()



    latest_entry = SquatDatatest.objects.latest('id')



    json_data0 = latest_entry.squat_count
    json_data1 = latest_entry.score
    json_data2 = latest_entry.accuracy
    return JsonResponse({'json_data0':json_data0, 'json_data1':json_data1, 'json_data2':json_data2 })