import joblib
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def DNN_2D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 원-핫 인코딩
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # 텐서플로우 랜덤 시드 설정
    tf.random.set_seed(66)
    initializer = tf.keras.initializers.GlorotUniform(seed=66)

    # DNN 모델 생성
    dnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(9,), name='angles'),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(7, activation='softmax', kernel_initializer=initializer),
    ])

    # 모델 학습 과정 설정
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # loss가 10번이상 떨어지면 과적합 방지를 위해 학습중지
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='min')

    # DNN 모델 학습
    history = dnn_model.fit(np.array(X_train), np.array(y_train), epochs=1000, callbacks=[early_stopping])

    # 손실률, 정확도 출력
    # train_loss, train_accuracy = dnn_model.evaluate(X_train, y_train)
    test_loss, test_accuracy = dnn_model.evaluate(X_test, y_test)
    # print(f'Train data Accuracy: {train_accuracy:.3f}')
    print(f'Test data Accuracy: {test_accuracy:.3f}')

    # 모델 저장
    dnn_model.save(model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = tf.keras.models.load_model(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 원-핫 인코딩
                y_new = tf.keras.utils.to_categorical(y_new)

                # 정확도 출력
                new_loss, new_accuracy = loaded_model.evaluate(X_new, y_new)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def DNN_2D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y',
                    'right_hip_x', 'right_hip_y',
                    'left_hip_x', 'left_hip_y',
                    'right_knee_x', 'right_knee_y',
                    'left_knee_x', 'left_knee_y',
                    'right_ankle_x', 'right_ankle_y',
                    'left_ankle_x', 'left_ankle_y']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 원-핫 인코딩
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # 텐서플로우 랜덤 시드 설정
    tf.random.set_seed(66)
    initializer = tf.keras.initializers.GlorotUniform(seed=66)

    # DNN 모델 생성
    dnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(16,), name='points'),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(7, activation='softmax', kernel_initializer=initializer),
    ])

    # 모델 학습 과정 설정
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # loss가 10번이상 떨어지면 과적합 방지를 위해 학습중지
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='min')

    # DNN 모델 학습
    history = dnn_model.fit(np.array(X_train), np.array(y_train), epochs=1000, callbacks=[early_stopping])

    # 손실률, 정확도 출력
    # train_loss, train_accuracy = dnn_model.evaluate(X_train, y_train)
    test_loss, test_accuracy = dnn_model.evaluate(X_test, y_test)
    # print(f'Train data Accuracy: {train_accuracy:.3f}')
    print(f'Test data Accuracy: {test_accuracy:.3f}')

    # 모델 저장
    dnn_model.save(model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = tf.keras.models.load_model(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y',
                                  'left_shoulder_x', 'left_shoulder_y',
                                  'right_hip_x', 'right_hip_y',
                                  'left_hip_x', 'left_hip_y',
                                  'right_knee_x', 'right_knee_y',
                                  'left_knee_x', 'left_knee_y',
                                  'right_ankle_x', 'right_ankle_y',
                                  'left_ankle_x', 'left_ankle_y']]
                y_new = new_data['label']

                # 원-핫 인코딩
                y_new = tf.keras.utils.to_categorical(y_new)

                # 정확도 출력
                new_loss, new_accuracy = loaded_model.evaluate(X_new, y_new)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def DNN_3D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 원-핫 인코딩
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # 텐서플로우 랜덤 시드 설정
    tf.random.set_seed(66)
    initializer = tf.keras.initializers.GlorotUniform(seed=66)

    # DNN 모델 생성
    dnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(9,), name='angles'),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(7, activation='softmax', kernel_initializer=initializer),
    ])

    # 모델 학습 과정 설정
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # loss가 10번이상 떨어지면 과적합 방지를 위해 학습중지
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='min')

    # DNN 모델 학습
    history = dnn_model.fit(np.array(X_train), np.array(y_train), epochs=1000, callbacks=[early_stopping])

    # 손실률, 정확도 출력
    # train_loss, train_accuracy = dnn_model.evaluate(X_train, y_train)
    test_loss, test_accuracy = dnn_model.evaluate(X_test, y_test)
    # print(f'Train data Accuracy: {train_accuracy:.3f}')
    print(f'Test data Accuracy: {test_accuracy:.3f}')

    # 모델 저장
    dnn_model.save(model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = tf.keras.models.load_model(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 원-핫 인코딩
                y_new = tf.keras.utils.to_categorical(y_new)

                # 정확도 출력
                new_loss, new_accuracy = loaded_model.evaluate(X_new, y_new)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def DNN_3D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                    'right_hip_x', 'right_hip_y', 'right_hip_z',
                    'left_hip_x', 'left_hip_y', 'left_hip_z',
                    'right_knee_x', 'right_knee_y', 'right_knee_z',
                    'left_knee_x', 'left_knee_y', 'left_knee_z',
                    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                    'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 원-핫 인코딩
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # 텐서플로우 랜덤 시드 설정
    tf.random.set_seed(66)
    initializer = tf.keras.initializers.GlorotUniform(seed=66)

    # DNN 모델 생성
    dnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(24,), name='points'),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(7, activation='softmax', kernel_initializer=initializer),
    ])

    # 모델 학습 과정 설정
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # loss가 5번이상 떨어지면 과적합 방지를 위해 학습중지
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='min')

    # DNN 모델 학습
    history = dnn_model.fit(np.array(X_train), np.array(y_train), epochs=1000, callbacks=[early_stopping])

    # 손실률, 정확도 출력
    # train_loss, train_accuracy = dnn_model.evaluate(X_train, y_train)
    test_loss, test_accuracy = dnn_model.evaluate(X_test, y_test)
    # print(f'Train data Accuracy: {train_accuracy:.3f}')
    print(f'Test data Accuracy: {test_accuracy:.3f}')

    # 모델 저장
    dnn_model.save(model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = tf.keras.models.load_model(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                  'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                                  'right_hip_x', 'right_hip_y', 'right_hip_z',
                                  'left_hip_x', 'left_hip_y', 'left_hip_z',
                                  'right_knee_x', 'right_knee_y', 'right_knee_z',
                                  'left_knee_x', 'left_knee_y', 'left_knee_z',
                                  'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
                y_new = new_data['label']

                # 원-핫 인코딩
                y_new = tf.keras.utils.to_categorical(y_new)

                # 정확도 출력
                new_loss, new_accuracy = loaded_model.evaluate(X_new, y_new)
                print(f'New data Accuracy: {new_accuracy:.3f}')


def RF_2D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 생성
    rf_model = RandomForestClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    rf_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = rf_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(rf_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def RF_2D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y',
                    'right_hip_x', 'right_hip_y',
                    'left_hip_x', 'left_hip_y',
                    'right_knee_x', 'right_knee_y',
                    'left_knee_x', 'left_knee_y',
                    'right_ankle_x', 'right_ankle_y',
                    'left_ankle_x', 'left_ankle_y']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 생성
    rf_model = RandomForestClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    rf_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = rf_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(rf_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y',
                                  'left_shoulder_x', 'left_shoulder_y',
                                  'right_hip_x', 'right_hip_y',
                                  'left_hip_x', 'left_hip_y',
                                  'right_knee_x', 'right_knee_y',
                                  'left_knee_x', 'left_knee_y',
                                  'right_ankle_x', 'right_ankle_y',
                                  'left_ankle_x', 'left_ankle_y']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def RF_3D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 생성
    rf_model = RandomForestClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    rf_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = rf_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(rf_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def RF_3D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                    'right_hip_x', 'right_hip_y', 'right_hip_z',
                    'left_hip_x', 'left_hip_y', 'left_hip_z',
                    'right_knee_x', 'right_knee_y', 'right_knee_z',
                    'left_knee_x', 'left_knee_y', 'left_knee_z',
                    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                    'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 생성
    rf_model = RandomForestClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    rf_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = rf_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(rf_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                  'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                                  'right_hip_x', 'right_hip_y', 'right_hip_z',
                                  'left_hip_x', 'left_hip_y', 'left_hip_z',
                                  'right_knee_x', 'right_knee_y', 'right_knee_z',
                                  'left_knee_x', 'left_knee_y', 'left_knee_z',
                                  'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')


def XGB_2D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGB 모델 생성
    xgb_model = XGBClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    xgb_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = xgb_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(xgb_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def XGB_2D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y',
                    'right_hip_x', 'right_hip_y',
                    'left_hip_x', 'left_hip_y',
                    'right_knee_x', 'right_knee_y',
                    'left_knee_x', 'left_knee_y',
                    'right_ankle_x', 'right_ankle_y',
                    'left_ankle_x', 'left_ankle_y']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGB 모델 생성
    xgb_model = XGBClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    xgb_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = xgb_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(xgb_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y',
                                  'left_shoulder_x', 'left_shoulder_y',
                                  'right_hip_x', 'right_hip_y',
                                  'left_hip_x', 'left_hip_y',
                                  'right_knee_x', 'right_knee_y',
                                  'left_knee_x', 'left_knee_y',
                                  'right_ankle_x', 'right_ankle_y',
                                  'left_ankle_x', 'left_ankle_y']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def XGB_3D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGB 모델 생성
    xgb_model = XGBClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    xgb_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = xgb_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(xgb_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def XGB_3D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                    'right_hip_x', 'right_hip_y', 'right_hip_z',
                    'left_hip_x', 'left_hip_y', 'left_hip_z',
                    'right_knee_x', 'right_knee_y', 'right_knee_z',
                    'left_knee_x', 'left_knee_y', 'left_knee_z',
                    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                    'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGB 모델 생성
    xgb_model = XGBClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    xgb_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = xgb_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(xgb_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                  'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                                  'right_hip_x', 'right_hip_y', 'right_hip_z',
                                  'left_hip_x', 'left_hip_y', 'left_hip_z',
                                  'right_knee_x', 'right_knee_y', 'right_knee_z',
                                  'left_knee_x', 'left_knee_y', 'left_knee_z',
                                  'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')


def LGBM_2D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LGBM 모델 생성
    lgbm_model = LGBMClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    lgbm_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = lgbm_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(lgbm_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def LGBM_2D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y',
                    'right_hip_x', 'right_hip_y',
                    'left_hip_x', 'left_hip_y',
                    'right_knee_x', 'right_knee_y',
                    'left_knee_x', 'left_knee_y',
                    'right_ankle_x', 'right_ankle_y',
                    'left_ankle_x', 'left_ankle_y']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LGBM 모델 생성
    lgbm_model = LGBMClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    lgbm_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = lgbm_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(lgbm_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y',
                                  'left_shoulder_x', 'left_shoulder_y',
                                  'right_hip_x', 'right_hip_y',
                                  'left_hip_x', 'left_hip_y',
                                  'right_knee_x', 'right_knee_y',
                                  'left_knee_x', 'left_knee_y',
                                  'right_ankle_x', 'right_ankle_y',
                                  'left_ankle_x', 'left_ankle_y']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def LGBM_3D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LGBM 모델 생성
    lgbm_model = LGBMClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    lgbm_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = lgbm_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(lgbm_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def LGBM_3D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                    'right_hip_x', 'right_hip_y', 'right_hip_z',
                    'left_hip_x', 'left_hip_y', 'left_hip_z',
                    'right_knee_x', 'right_knee_y', 'right_knee_z',
                    'left_knee_x', 'left_knee_y', 'left_knee_z',
                    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                    'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LGBM 모델 생성
    lgbm_model = LGBMClassifier(n_estimators=200, random_state=66)

    # 모델 학습
    lgbm_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = lgbm_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(lgbm_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                  'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                                  'right_hip_x', 'right_hip_y', 'right_hip_z',
                                  'left_hip_x', 'left_hip_y', 'left_hip_z',
                                  'right_knee_x', 'right_knee_y', 'right_knee_z',
                                  'left_knee_x', 'left_knee_y', 'left_knee_z',
                                  'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')


def LR_2D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 로지스틱 회귀 모델 생성
    lr_model = LogisticRegression(max_iter=1000, random_state=66)

    # 모델 학습
    lr_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = lr_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(lr_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def LR_2D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y',
                    'right_hip_x', 'right_hip_y',
                    'left_hip_x', 'left_hip_y',
                    'right_knee_x', 'right_knee_y',
                    'left_knee_x', 'left_knee_y',
                    'right_ankle_x', 'right_ankle_y',
                    'left_ankle_x', 'left_ankle_y']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 로지스틱 회귀 모델 생성
    lr_model = LogisticRegression(max_iter=1000, random_state=66)

    # 모델 학습
    lr_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = lr_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(lr_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y',
                                  'left_shoulder_x', 'left_shoulder_y',
                                  'right_hip_x', 'right_hip_y',
                                  'left_hip_x', 'left_hip_y',
                                  'right_knee_x', 'right_knee_y',
                                  'left_knee_x', 'left_knee_y',
                                  'right_ankle_x', 'right_ankle_y',
                                  'left_ankle_x', 'left_ankle_y']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def LR_3D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 로지스틱 회귀 모델 생성
    lr_model = LogisticRegression(max_iter=1000, random_state=66)

    # 모델 학습
    lr_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = lr_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(lr_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def LR_3D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                    'right_hip_x', 'right_hip_y', 'right_hip_z',
                    'left_hip_x', 'left_hip_y', 'left_hip_z',
                    'right_knee_x', 'right_knee_y', 'right_knee_z',
                    'left_knee_x', 'left_knee_y', 'left_knee_z',
                    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                    'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 로지스틱 회귀 모델 생성
    lr_model = LogisticRegression(max_iter=1000, random_state=66)

    # 모델 학습
    lr_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = lr_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(lr_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                  'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                                  'right_hip_x', 'right_hip_y', 'right_hip_z',
                                  'left_hip_x', 'left_hip_y', 'left_hip_z',
                                  'right_knee_x', 'right_knee_y', 'right_knee_z',
                                  'left_knee_x', 'left_knee_y', 'left_knee_z',
                                  'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')


def GNB_2D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 나이브 베이즈 모델 생성
    gnb_model = GaussianNB()

    # 모델 학습
    gnb_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = gnb_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(gnb_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def GNB_2D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y',
                    'right_hip_x', 'right_hip_y',
                    'left_hip_x', 'left_hip_y',
                    'right_knee_x', 'right_knee_y',
                    'left_knee_x', 'left_knee_y',
                    'right_ankle_x', 'right_ankle_y',
                    'left_ankle_x', 'left_ankle_y']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 나이브 베이즈 모델 생성
    gnb_model = GaussianNB()

    # 모델 학습
    gnb_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = gnb_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(gnb_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y',
                                  'left_shoulder_x', 'left_shoulder_y',
                                  'right_hip_x', 'right_hip_y',
                                  'left_hip_x', 'left_hip_y',
                                  'right_knee_x', 'right_knee_y',
                                  'left_knee_x', 'left_knee_y',
                                  'right_ankle_x', 'right_ankle_y',
                                  'left_ankle_x', 'left_ankle_y']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def GNB_3D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    #'knee_knee_dis'
                    ]]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 나이브 베이즈 모델 생성
    gnb_model = GaussianNB()

    # 모델 학습
    gnb_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = gnb_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(gnb_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  #'knee_knee_dis'
                                  ]]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def GNB_3D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                    'right_hip_x', 'right_hip_y', 'right_hip_z',
                    'left_hip_x', 'left_hip_y', 'left_hip_z',
                    'right_knee_x', 'right_knee_y', 'right_knee_z',
                    'left_knee_x', 'left_knee_y', 'left_knee_z',
                    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                    'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 나이브 베이즈 모델 생성
    gnb_model = GaussianNB()

    # 모델 학습
    gnb_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = gnb_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(gnb_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                  'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                                  'right_hip_x', 'right_hip_y', 'right_hip_z',
                                  'left_hip_x', 'left_hip_y', 'left_hip_z',
                                  'right_knee_x', 'right_knee_y', 'right_knee_z',
                                  'left_knee_x', 'left_knee_y', 'left_knee_z',
                                  'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')


def DT_2D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 의사결정 트리 모델 생성
    dt_model = DecisionTreeClassifier(random_state=66)

    # 모델 학습
    dt_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = dt_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(dt_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def DT_2D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y',
                    'right_hip_x', 'right_hip_y',
                    'left_hip_x', 'left_hip_y',
                    'right_knee_x', 'right_knee_y',
                    'left_knee_x', 'left_knee_y',
                    'right_ankle_x', 'right_ankle_y',
                    'left_ankle_x', 'left_ankle_y']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 의사결정 트리 모델 생성
    dt_model = DecisionTreeClassifier(random_state=66)

    # 모델 학습
    dt_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = dt_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(dt_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y',
                                  'left_shoulder_x', 'left_shoulder_y',
                                  'right_hip_x', 'right_hip_y',
                                  'left_hip_x', 'left_hip_y',
                                  'right_knee_x', 'right_knee_y',
                                  'left_knee_x', 'left_knee_y',
                                  'right_ankle_x', 'right_ankle_y',
                                  'left_ankle_x', 'left_ankle_y']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def DT_3D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 의사결정 트리 모델 생성
    dt_model = DecisionTreeClassifier(random_state=66)

    # 모델 학습
    dt_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = dt_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(dt_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def DT_3D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                    'right_hip_x', 'right_hip_y', 'right_hip_z',
                    'left_hip_x', 'left_hip_y', 'left_hip_z',
                    'right_knee_x', 'right_knee_y', 'right_knee_z',
                    'left_knee_x', 'left_knee_y', 'left_knee_z',
                    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                    'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 의사결정 트리 모델 생성
    dt_model = DecisionTreeClassifier(random_state=66)

    # 모델 학습
    dt_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = dt_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(dt_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                  'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                                  'right_hip_x', 'right_hip_y', 'right_hip_z',
                                  'left_hip_x', 'left_hip_y', 'left_hip_z',
                                  'right_knee_x', 'right_knee_y', 'right_knee_z',
                                  'left_knee_x', 'left_knee_y', 'left_knee_z',
                                  'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')


def SVC_2D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVC 모델 생성
    svc_model = SVC(kernel='rbf', random_state=66, probability=True)

    # 모델 학습
    svc_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = svc_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(svc_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def SVC_2D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y',
                    'right_hip_x', 'right_hip_y',
                    'left_hip_x', 'left_hip_y',
                    'right_knee_x', 'right_knee_y',
                    'left_knee_x', 'left_knee_y',
                    'right_ankle_x', 'right_ankle_y',
                    'left_ankle_x', 'left_ankle_y']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVC 모델 생성
    svc_model = SVC(kernel='rbf', random_state=66, probability=True)

    # 모델 학습
    svc_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = svc_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(svc_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y',
                                  'left_shoulder_x', 'left_shoulder_y',
                                  'right_hip_x', 'right_hip_y',
                                  'left_hip_x', 'left_hip_y',
                                  'right_knee_x', 'right_knee_y',
                                  'left_knee_x', 'left_knee_y',
                                  'right_ankle_x', 'right_ankle_y',
                                  'left_ankle_x', 'left_ankle_y']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def SVC_3D_angle(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['back_angle_R', 'back_angle_L',
                    'knee_angle_R', 'knee_angle_L',
                    'ankle_knee_knee_R', 'ankle_knee_knee_L',
                    'hip_hip_knee_R', 'hip_hip_knee_L',
                    'knee_knee_dis']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVC 모델 생성
    svc_model = SVC(kernel='rbf', random_state=66, probability=True)

    # 모델 학습
    svc_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = svc_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(svc_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['back_angle_R', 'back_angle_L',
                                  'knee_angle_R', 'knee_angle_L',
                                  'ankle_knee_knee_R', 'ankle_knee_knee_L',
                                  'hip_hip_knee_R', 'hip_hip_knee_L',
                                  'knee_knee_dis']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')

def SVC_3D_point(train_route, model_save_route, test_base_route, do_eval=False):
    # train CSV 파일 불러오기
    data_train = pd.read_csv(train_route)
    print(len(data_train))

    # 공백이 있는 행 제거
    data_train = data_train.dropna()

    # x_train과 y_train 추출
    X = data_train[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                    'right_hip_x', 'right_hip_y', 'right_hip_z',
                    'left_hip_x', 'left_hip_y', 'left_hip_z',
                    'right_knee_x', 'right_knee_y', 'right_knee_z',
                    'left_knee_x', 'left_knee_y', 'left_knee_z',
                    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                    'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    y = data_train['label']

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVC 모델 생성
    svc_model = SVC(kernel='rbf', random_state=66, probability=True)

    # 모델 학습
    svc_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = svc_model.predict(X_test)

    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Train data Accuracy: {accuracy:.3f}')

    # 모델 저장
    joblib.dump(svc_model, model_save_route)

    if do_eval:
        for i in range(1, 4):
            for j in range(1, 4):
                test_csv = f'test{i}_{j}.csv'
                test_full_path = test_base_route + test_csv

                # 저장된 모델 불러오기
                loaded_model = joblib.load(model_save_route)

                # 새로운 데이터 불러오기
                new_data = pd.read_csv(test_full_path)

                # x_new 추출
                X_new = new_data[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                  'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                                  'right_hip_x', 'right_hip_y', 'right_hip_z',
                                  'left_hip_x', 'left_hip_y', 'left_hip_z',
                                  'right_knee_x', 'right_knee_y', 'right_knee_z',
                                  'left_knee_x', 'left_knee_y', 'left_knee_z',
                                  'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
                y_new = new_data['label']

                # 새로운 데이터에 대한 예측
                new_predictions = loaded_model.predict(X_new)

                # 정확도 출력
                new_accuracy = accuracy_score(y_new, new_predictions)
                print(f'New data Accuracy: {new_accuracy:.3f}')