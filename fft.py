
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import os
import sys
import io
import cv2
import h5py
import keyboard
# function to convert recorded AES key
def convert_plaintext_key(file_path, key_num):
    try:
        pks = np.loadtxt(file_path, dtype=np.int16, delimiter = ',')
    except:
        print("Error2: can't open txt file for read")
        sys.exit(-1)
    print(pks.shape)
    plaintext = np.zeros((key_num, 16), dtype=np.int16)
    aes_key = np.zeros((key_num, 16), dtype=np.int16)
    for n in range(int(len(pks)/2)):
        plaintext[n] = pks[2*n+1]
        aes_key[n] = pks[2*n]
    return plaintext, aes_key

def get_avarage(array1, array2, num):
    if (len(array1)!= len(array2)):
        print("Error3: lengths of arrays are different.")
        sys.exit(-1)
    else:
        for n in range(len(array1)):
            array1[n] = (array1[n] * num + array2[n]) / (num + 1)

        return array1

#convert .raw to complex
def convert_raw_to_dataset(traces_file_path, train_group, valid_group,  key_num, traces_per_key, frequency, length_signal, validaion_rate):
    recordLen = length_signal*2  # number of int16's per record
    recordSize = recordLen * 2  # size of a record in bytes
    image = np.dtype((np.int16, (500,500)))
    train_traces = np.zeros(int(key_num*(1.0-validaion_rate)), dtype=image)
    validation_traces = np.zeros(int(key_num*validaion_rate), dtype=image)
    with open(traces_file_path, 'rb') as file:
        # collect data of train set
        ###
        print("processing train trace")
        ###
        for i in range(0, int(key_num*(1.0-validaion_rate))):
            avarage = np.zeros(recordLen, dtype=np.int16)
            for k in range(0, traces_per_key):
                recordNo = i*traces_per_key + k  # the index of a target record in the file
                # Reading a record recordNo from file into the memArray
                file.seek(recordSize * recordNo)
                bytes = file.read(recordSize)
                memArray = np.frombuffer(bytes, dtype=np.int16).copy()
                avarage = get_avarage(avarage, memArray, k)

            img = avarage[0::2] + 1j * avarage[1::2]
            data = fft_fuc(img, frequency, length_signal)
            print(data.shape)
            train_traces[i]= data
        # collect data of validation set
        ###
        print("processing valid trace")
        ###
        for i in range(int(key_num*(1.0-validaion_rate)), key_num):
            avarage = np.zeros(recordLen, dtype=np.int16)
            for k in range(0, traces_per_key):
                recordNo = i*traces_per_key + k  # the index of a target record in the file
                # Reading a record recordNo from file into the memArray
                file.seek(recordSize * recordNo)
                bytes = file.read(recordSize)
                memArray = np.frombuffer(bytes, dtype=np.int16).copy()
                avarage = get_avarage(avarage, memArray, k)

            img = avarage[0::2] + 1j * avarage[1::2]
            data = fft_fuc(img, frequency, length_signal)
            print(data.shape)
            validation_traces[i-int(key_num*(1.0-validaion_rate))]= data
    file.close()

    train_group.create_dataset(name = "trace", data = train_traces, dtype = train_traces.dtype)
    valid_group.create_dataset(name = "trace", data = validation_traces, dtype= validation_traces.dtype)
# img to nparray
def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img
#fft
def fft_fuc(signal, frequency, length_signal):
    Fs = frequency
    T =1/Fs
    L = length_signal
    t =np.zeros(L-1)
    for n in range(L-1):
        t[n] = n*Fs/L
    half_t = t[range(int(L/2))]
    fft_y = fft(signal)
    fft_y =fftshift(fft_y)
    abs_y = np.abs(fft_y)
    normalization_half_y = abs_y[range(int(L / 2))]
    '''
    fft_y = np.abs(fft_y)
    fft_y /= max(fft_y)
    fft_x = fftfreq(len(fft_fft_y), 1 / len(fft_fft_y))
    '''
    fig = plt.figure(figsize=(5.0,5.0))
    #plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    #plt.plot(fftshift(fft_x), fftshift(fft_y))
    plt.plot(half_t,normalization_half_y)
    plot_img_np = get_img_from_fig(fig)
    plt.close(fig)
    return plot_img_np

# compute hanmming weight of plaintext and key
def labelize(plaintext, key):
    return np.int16(plaintext ^ key)

if __name__ == "__main__":
    if len(sys.argv)!=2:
        file_path = 'train.raw'
        SAMPLE_RATE = 10000000
        frequency = 80000000
        key_num = 100
        traces_per_key = 100
        record_time = 12048
        key_file_path = "train_aes_key.txt"
        validation_rate = 0.3
    else:
        file_path = 'train.raw'
        SAMPLE_RATE = 10000000
        frequency = 80000000
        key_num = 100
        traces_per_key = 100
        record_time = 12048
        validation_rate = 0.3
        #todo : read_parameter

    sample_num = key_num * traces_per_key
    plaintext, aes_key = convert_plaintext_key(key_file_path, key_num)
    length_signal = int(SAMPLE_RATE * record_time/1000) // sample_num

    saving_path = "AES_key_recover_train.hdf5"

    try:
        output_file = h5py.File(saving_path, 'w')
    except:
        print("Error1:can't create HDF5 file")
        sys.exit(-1)

    train_set_group = output_file.create_group("train_set")
    valid_set_group = output_file.create_group("valid_set")

    convert_raw_to_dataset(file_path, train_set_group, valid_set_group, key_num, traces_per_key, frequency, length_signal, validation_rate)
    ###
    print("Computing labels")
    ###
    train_index = [n for n in range(0, int(key_num*(1.0 - validation_rate)))]
    valid_index = [n for n in range(int(key_num*(1.0 - validation_rate)), key_num)]
    labels_train = labelize(plaintext[train_index], aes_key[train_index])
    labels_valid = labelize(plaintext[valid_index], aes_key[valid_index])

    ###
    print("Creating output file")
    ###

    train_set_group.create_dataset(name="labels", data=labels_train, dtype=labels_train.dtype)
    valid_set_group.create_dataset(name="labels", data=labels_valid, dtype=labels_valid.dtype)

    metadata_type = np.dtype([("plaintext", plaintext.dtype, (len(plaintext[0]),)),
                              ("key", aes_key.dtype, (len(aes_key[0]),))
                              ])
    train_metadata = np.array([(plaintext[n], aes_key[n]) for n in zip(train_index)], dtype=metadata_type)
    valid_metadata = np.array([(plaintext[n], aes_key[n]) for n in zip(valid_index)], dtype=metadata_type)

    train_set_group.create_dataset(name="metadata", data=train_metadata, dtype=metadata_type)
    valid_set_group.create_dataset(name="metadata", data=valid_metadata, dtype=metadata_type)

    output_file.flush()
    output_file.close()
    try:
        input("Press enter to exit ....")

    except SyntaxError:
        pass