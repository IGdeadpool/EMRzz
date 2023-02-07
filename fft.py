
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
file_path = 'test2.raw'
data = np.fromfile(file_path, dtype=np.int16) #1-D array
print(data.shape)
SAMPLE_RATE = 10000000
example_num = 100
fft_size = example_num*100
length = len(data)//(fft_size*2)

for i in (0, int(fft_size*0.7)):
    f = open(file_path, 'rb')
    complex_part = np.fromfile(f, count=length*2, dytype=np.int16, offset=2*i*(length*2))
    complex_num = complex_part[0::2] + 1j * complex_part[1::2]
    fft_fft_y = fft(complex_num[i])
    print(fft_fft_y.shape)
    fft_y = np.abs(fft_fft_y)
    fft_y /= max(fft_y)
    fft_x = fftfreq(len(fft_fft_y), 1 / len(fft_fft_y))
    plt.figure()
    plt.plot(fftshift(fft_x), fftshift(fft_y))
    plt.savefig('./train_set/{}.png'.format(i + 1), bbox_inches='tight')
for i in range(int(fft_size*0.7), fft_size):
    f = open(file_path, 'rb')
    complex_part = np.fromfile(f, count=length * 2, dytype=np.int16, offset=2 * i * (length * 2))
    complex_num = complex_part[0::2] + 1j * complex_part[1::2]
    fft_fft_y = fft(complex_num[i])
    print(fft_fft_y.shape)
    fft_y = np.abs(fft_fft_y)
    fft_y /= max(fft_y)
    fft_x = fftfreq(len(fft_fft_y), 1 / len(fft_fft_y))
    plt.figure()
    plt.plot(fftshift(fft_x), fftshift(fft_y))
    plt.savefig('./train_set/{}.png'.format(i + 1), bbox_inches='tight')
















 ###########
complex_num = data[0::2] + 1j * data[1::2]
print(complex_num.shape)
complex_num = complex_num.reshape(example_num, length)
for i in range(0,example_num*0.7):
    fft_fft_y = fft(complex_num[i])
    print(fft_fft_y.shape)
    fft_y = np.abs(fft_fft_y)
    fft_y/=max(fft_y)
    fft_x = fftfreq(len(fft_fft_y), 1/len(fft_fft_y))
    plt.figure()
    plt.plot(fftshift(fft_x), fftshift(fft_y))
    plt.savefig('./train_set/{}.png'.format(i + 1) , bbox_inches='tight')
for i in range(example_num * 0.7, example_num):
    fft_fft_y = fft(complex_num[i])
    print(fft_fft_y.shape)
    fft_y = np.abs(fft_fft_y)
    fft_y /= max(fft_y)
    fft_x = fftfreq(len(fft_fft_y), 1 / len(fft_fft_y))
    plt.figure()
    plt.plot(fftshift(fft_x), fftshift(fft_y))
    plt.savefig('./test_set/{}.png'.format(i + 1), bbox_inches='tight')

 ###