import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import MultipleLocator
from scipy.fft import fft, fftfreq, fftshift
import os
import sys
import io
import cv2
import h5py
def read_and_fft(file, points_to_read, num_to_seek):
    file.seek((num_to_seek) * points_to_read)
    bytes = file.read(points_to_read)
    memArray = np.frombuffer(bytes, dtype='<f4').copy()
    trace = memArray[0::2] + 1j * memArray[1::2]
    fft_energy = fft(trace)
    x = fftfreq(points_to_read//8, d=1/80000000)

    plt.figure()
    plt.plot(x, abs(fft_energy))
    plt.show()
if __name__ == "__main__":
    sample_rate = 10000000
    real_key_num = 1000
    key_num = 1000
    sample_pre_key = 100
    file_path = "aes_1000.raw"
    time_record = 120897
    with open(file_path, 'rb') as file:
        sample_energy = np.zeros((key_num,sample_pre_key), dtype='f4')
        for t in range (key_num):
            points_to_read = int(time_record * sample_rate / (real_key_num * 1000 * sample_pre_key)) * 8
            for i in range(sample_pre_key):
                file.seek((i + t * sample_pre_key) * points_to_read)
                bytes = file.read(points_to_read)
                memArray = np.frombuffer(bytes, dtype='<f4').copy()
                trace = memArray[0::2] + 1j * memArray[1::2]
                fft_energy = fft(trace)
                energy = np.square(np.abs(fft_energy))
                energy_sum = np.sum(energy)
                sample_energy[t][i] = energy_sum
            # sample_energy[t]/=np.max(sample_energy[t])
        #sample_energy = sample_energy / np.max(sample_energy)
        print(sample_energy.shape)
        samples = np.linspace(0, sample_pre_key, sample_pre_key)
        samples.shape = (sample_pre_key,1)
        sample_energy.shape = (key_num, sample_pre_key)

        ax = plt.gca()
        #ax.yaxis.set_major_locator(MultipleLocator(np.max(sample_energy) / 0.2))
        #ax.xaxis.set_major_locator(MultipleLocator(400 / 50))
        plt.figure(figsize=(19.2,10.8))
        for i in range (key_num):
            tempsample = sample_energy[i]
            tempsample.shape = (sample_pre_key, 1)
            plt.plot(samples, tempsample)
        plt.savefig('1000_10_aes_1000-sample_plot.jpg')
        plt.show()
        """points_to_read = int(time_record * sample_rate / (real_key_num * 1000 * sample_pre_key)) * 8
        for i in [0, 100, 500, 1000, 2000, 4000]:
            read_and_fft(file, points_to_read, i)"""