import play_test_modules

# GaussianNB, 2D, angle
test_video_path = '../data/test/video/test1_1.mp4'
model_path = '../models/gnb_model_2D_angle.joblib'

play_test_modules.play_2D_angle(test_video_path, model_path)

# # GaussianNB, 2D, point
# test_video_path = '../data/test/video/test1_1.mp4'
# model_path = '../models/gnb_model_2D_point.joblib'
#
# play_test_modules.play_2D_point(test_video_path, model_path)

# # GaussianNB, 3D, angle
# test_video_path = '../data/test/video/test1_1.mp4'
# model_path = '../models/gnb_model_3D_angle.joblib'
#
# play_test_modules.play_3D_angle(test_video_path, model_path)

# # GaussianNB, 3D, point
# test_video_path = '../data/test/video/test1_1.mp4'
# model_path = '../models/gnb_model_3D_point.joblib'
#
# play_test_modules.play_3D_point(test_video_path, model_path)
