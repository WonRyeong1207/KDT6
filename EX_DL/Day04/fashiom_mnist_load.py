# URL기반 데이터 파일 저장 코드 -----------------------------------------------------

from urllib.request import urlretrieve

TEST_URL  = 'https://media.githubusercontent.com/media/fpleoni/fashion_mnist/master/fashion-mnist_test.csv'
TRAIN_URL = 'https://media.githubusercontent.com/media/fpleoni/fashion_mnist/master/fashion-mnist_train.csv'

TRAIN_FILE = '../data/fashion-mnist_train.csv'
TEST_FILE  = '../data/fashion-mnist_test.csv'
# 데이터 파일로 저장
urlretrieve(TRAIN_URL, TRAIN_FILE)
urlretrieve(TEST_URL, TEST_FILE)