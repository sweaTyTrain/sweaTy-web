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

