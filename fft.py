
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import cv2
import os
from PIL import Image

def make_name(img_path, img_file_txt, label):
    with open(img_file_txt, 'a+') as fp:
        fp.write(img_path)
        fp.write('  ')
        fp.write(label)
        fp.write('\n')
def tentotwo(num):
    s = []
    binstring = ''
    while num >0:
        rem = num%2
        s.append(rem)
        number = num //2
    while len(s)>0:
        binstring = binstring + str(s.pop())
    return (binstring)

def SIFT(img):
    sift = cv2.xfeature2d.SIFT_create()
    kp,des = sift.detectAndCompute(img, None)
    size = len(kp)
    return des, size
def match(des1, des2, size):
    match = True
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if len(good)>int(0.7*size):
        match = False
    return match

file_path = 'test2.raw'
data = np.fromfile(file_path, dtype=np.int16) #1-D array
print(data.shape)
SAMPLE_RATE = 10000000
example_num = 100
slot = 100
fft_size = example_num*slot
length = len(data)//(fft_size*2)
train_set_count = 0
test_set_count = 0
aes_key = np.loadtxt("100_aes.txt", dtype='str', delimiter=' ')
with open("train_set.txt", 'a+') as train:
    train.truncate(0)
with open("test_set.txt", 'a+') as test:
    test.truncate(0)

for i in range(0, example_num):
    f = open(file_path, 'rb')
    complex_part_ref_img = np.fromfile(f, count=length * 2, dytype=np.int16, offset=2 * i * slot * (length * 2))
    ref_img = complex_part_ref_img[0::2] + 1j * complex_part_ref_img[1::2]
    fft_fft_y = fft(ref_img)
    fft_y = np.abs(fft_fft_y)
    fft_y /= max(fft_y)
    fft_x = fftfreq(len(fft_fft_y), 1 / len(fft_fft_y))
    plt.figure()
    img = plt.plot(fftshift(fft_x), fftshift(fft_y))
    des_ref_img, point_size = SIFT(img)
    train_set_count += 1
    plt.savefig('./train_set/{}.png'.format(train_set_count), bbox_inches='tight')
    root_name = '/root/PycharmProjects/pythonProject/train_set/' + f"{train_set_count}.png"
    make_name(root_name, "train_set.txt", tentotwo(aes_key[i],16))
    for n in (1, int(slot * 0.7)):

        complex_part = np.fromfile(f, count=length * 2, dytype=np.int16, offset=((2 * i * slot) + n) * (length * 2))
        complex_num = complex_part[0::2] + 1j * complex_part[1::2]
        fft_fft_y = fft(complex_num[i])
        print(fft_fft_y.shape)
        fft_y = np.abs(fft_fft_y)
        fft_y /= max(fft_y)
        fft_x = fftfreq(len(fft_fft_y), 1 / len(fft_fft_y))
        plt.figure()
        com_img = plt.plot(fftshift(fft_x), fftshift(fft_y))
        des_img, img_size = SIFT(com_img)
        if match(des_ref_img,des_img,point_size):
            train_set_count += 1
            plt.savefig('./train_set/{}.png'.format(train_set_count), bbox_inches='tight')
            root_name = '/root/PycharmProjects/pythonProject/train_set/' + f"{train_set_count}.png"
            make_name(root_name, "train_set.txt", tentotwo(aes_key[i], 16))
    for n in range(int(slot * 0.7), slot):
        complex_part = np.fromfile(f, count=length * 2, dytype=np.int16, offset=2 * i * (length * 2))
        complex_num = complex_part[0::2] + 1j * complex_part[1::2]
        fft_fft_y = fft(complex_num[i])
        print(fft_fft_y.shape)
        fft_y = np.abs(fft_fft_y)
        fft_y /= max(fft_y)
        fft_x = fftfreq(len(fft_fft_y), 1 / len(fft_fft_y))
        plt.figure()
        com_img = plt.plot(fftshift(fft_x), fftshift(fft_y))
        des_img, img_size = SIFT(com_img)
        if match(des_ref_img, des_img, point_size):
            test_set_count += 1
            plt.savefig('./test_set/{}.png'.format(test_set_count), bbox_inches='tight')
            root_name = '/root/PycharmProjects/pythonProject/test_set/' + f"{test_set_count}.png"
            make_name(root_name, "test_set.txt", tentotwo(aes_key[i], 16))