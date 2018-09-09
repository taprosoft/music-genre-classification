import numpy as np
import os

import keras
from keras.models import Model, load_model
import cv2

num_classes = 10

model_name = 'model_cnn_genre.h5'


CLASSES = np.array(['Cai Luong', 'Cach Mang', 'Dan Ca - Que Huong', 'Dance', 'Khong Loi',
               'Thieu Nhi', 'Trinh', 'Tru Tinh', 'Rap Viet', 'Rock Viet'])


rows = 128
cols = 1024
channels = 2
split_count = 10

def load_images(file_paths):
    img_arr = np.zeros((split_count * len(file_paths), rows, cols, channels), dtype='uint8')
    index = 0
    max_width = 10337

    for im_path in file_paths:
        if isinstance(im_path, list):
            im_stack = np.zeros((rows, max_width, channels), dtype='uint8')
            for i, im_name in enumerate(im_path):
                im = cv2.imread(im_name, 0)
                if im.shape[0] != rows:
                    im = cv2.resize(im ,(max_width, rows))
                if im.shape[1] > max_width:
                    im = im[:, :max_width]
                im_stack[:,:im.shape[1],i] = im
        else:
            im_stack = cv2.imread(im_path, 0)

        offset = 0
        for i in range(split_count):
            img_arr[index] = im_stack[:,offset:offset+cols]
            index += 1
            offset += cols

    return img_arr


def get_class_from_proba(y_proba):

    class_weights = np.ones(10)
    class_weights[7] = 0.6
    y_proba = np.multiply(y_proba, class_weights)

    num_frames = y_proba.shape[0]
    frame_weights = np.ones((num_frames,1))
    frame_weights[-2:,0] = [0.6, 0.6]

    y_proba = np.mean(y_proba * frame_weights, axis=0)
    print(y_proba)

    top_3 = y_proba.argsort()[-3:][::-1]
    print(top_3)
    res = top_3[0]

    return res


def predict_from_folder_spectr(spectr_folder, data_folder, output_file):
    from os import listdir
    from os.path import isfile, join

    # test_file_path = 'test.csv'
    # with open(test_file_path, 'r') as f:
    #     songs = [line.split() for line in f]

    output_file = open(output_file, 'w')
    output_file.write('Id, Genre\n')

    # spectr_folder = '/tmp/spectr/'
    songs = [s for s in listdir(data_folder) if s.endswith(".mp3")]

    model = load_model('/model/music_genre_cnn.h5')

    N = len(songs)
    for i, song in enumerate(songs):
        # song = song[
        print('Predicting song {} / {}: '.format(i + 1, N) + song)
        song_base = song.split('.')[0]
        im_paths = [[join(spectr_folder, song_base+'_h.png'), join(spectr_folder, song_base+'_p.png')]]
        print(im_paths)
        x_arr = load_images(im_paths)
        print(x_arr.shape)

        y_proba = model.predict(x_arr)
        res = get_class_from_proba(y_proba)

        genre = str(res + 1)

        line = song + ',' + str(genre) + '\n'

        print(line[:-1])
        output_file.write(line)

    output_file.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectr_dir', help='spectrogram directory')
    parser.add_argument('--song_dir', help='song directory')
    parser.add_argument('--output_file', help='output filename')
    args = parser.parse_args()

    predict_from_folder_spectr(args.spectr_dir, args.song_dir, args.output_file)