
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

file_path = 'test2.raw'
# 16 位 raw 数据
data = np.fromfile(file_path, dtype=np.int16) #1-D array
print(data.shape)
SAMPLE_RATE = 10000000
length = len(data)//200
for i in range(0, len(data)-200*length):
    data=np.delete(data, -1)
complex_num = data[0::2] + 1j * data[1::2]
print(complex_num.shape)
data1 = complex_num.reshape(100,length)
for i in range(0,100):
    fft_fft_y = fft(data1[i])
    print(fft_fft_y.shape)
    fft_y = np.abs(fft_fft_y)
    fft_y/=max(fft_y)
    fft_x = fftfreq(len(fft_fft_y), 1/len(fft_fft_y))
    plt.figure()
    plt.plot(fftshift(fft_x), fftshift(fft_y))
    plt.savefig('./img/pic-{}.png'.format(i + 1) , bbox_inches='tight')
