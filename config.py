import os

main_path = './dataset'
images_path = os.path.join(main_path, 'images')
test_path = os.path.join(main_path, 'test.csv')
train_path = os.path.join(main_path, 'train.csv')
sub_path = os.path.join(main_path, 'sample_submission.csv')

batch_size = 20
epochs = 20
input_shape = (273, 409, 3)
