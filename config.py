import os

main_path = './dataset'
images_path = os.path.join(main_path, 'images')
test_path = os.path.join(main_path, 'test.csv')
train_path = os.path.join(main_path, 'train.csv')
sub_path = os.path.join(main_path, 'sample_submission.csv')
models_path = os.path.join('.', 'models')

batch_size = 20
epochs = 60
# input_shape = (273, 409, 3)
input_shape = (224, 224, 3)