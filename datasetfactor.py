import numpy as np
import torch as tor
import matplotlib as plt
import os
from PIL import Image

def make_name(file_name_from, img_file_txt, label_array, length):
    photo_name_chr = os.listdir('./' + str(file_name_from))
    photo_name_chr.sort(key = lambda x: int(x))
    # print(photo_name_chr)
    with open(img_file_txt, 'w') as fp:
        for i in range(0,length):
            for n in range(100):
                root_name = 'C:/Users/User/Documents/EMRzz/' + str(file_name_from) + '/' + str(photo_name_chr[100*i+n+1])
                fp.write(root_name)
                fp.write('  ')
                fp.write(label_array[i])
                fp.write('\n')

def show_photo():
    with open('name.txt', 'r') as fp2:
        a = fp2.readline().split()
        # print(a[0])
        img = Image.open(a[0])
        img.show()

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

count = 100
aes_key = np.loadtxt("100_aes.txt", dtype='str', delimiter=' ')
print(aes_key)
train_set  = np.full(count*0.7, '0000000000000')
test_set = np.full(count*0.3, '0000000000000')
for i in range (count*0.7):
    train_set[i] = tentotwo(int(aes_key[i],16))
for i in range(count*0.7,count):
    test_set[i] = tentotwo(int(aes_key[i], 16))
make_name('train_folder', 'train_set', train_set, count*0.7)
make_name('test_folder', 'test_set.txt', test_set, count*0.3)

