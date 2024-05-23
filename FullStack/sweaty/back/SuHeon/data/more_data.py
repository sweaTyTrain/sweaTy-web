# 점 좌표에 노이즈를 추가해 데이터셋 늘리기 시도해봤음. 결과적으로는 이러한 방식으로 train셋을 늘려도 효과는 별로 없는듯

import pandas as pd
import numpy as np

# CSV 파일 불러오기
data_train = pd.read_csv("train_new_angle_dis_2D_0.csv")

for i in range(100):
    data_tmp = pd.read_csv("train_new_angle_dis_2D_0.csv")

    # 평균=0, 표준편차=0.05인 정규분포 난수값 더하기
    for column in data_tmp.columns:
        if column != 'label':  # 'label' 열은 레이블이므로 제외
            noise = np.round(np.random.normal(loc=0, scale=0.1, size=len(data_tmp)), 1)
            data_tmp[column] = data_tmp[column] + noise


    data_train = pd.concat([data_train, data_tmp])

print(data_train)
# print(np.random.normal(loc=0, scale=0.1, size=100))
# csv파일로 저장
data_train.to_csv("train_more_data.csv", index=False)