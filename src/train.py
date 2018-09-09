import numpy as np
import os
from tqdm import tqdm
from keras.optimizers import Adadelta
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
from util import plot_confusion_matrix
from config import args, rows, cols, channels, split_count, epochs, CLASSES, \
    num_classes, model_name, max_width, batch_size


def load_images(file_paths):
    img_arr = np.zeros((split_count * len(file_paths), rows, cols, channels), dtype='uint8')
    index = 0

    for im_path in tqdm(file_paths):
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


def load_data_from_spectrogram_dir(spectr_dir, spectr_types, label_csv = 'train.csv'):
    files_list = []
    label = []
    with open(label_csv, 'r') as f:
        songs = [line.strip().split(',') for line in f]
        for song in songs:
            files_list.append([os.path.join(spectr_dir, "{}_{}.png".format(song[0].split('.')[0], spectr_type)) for spectr_type in spectr_types])
            for i in range(split_count):
                label.append(int(song[1]) - 1)

    print('Got {} songs. Loading spectrogram...'.format(len(files_list)))
    x_train = load_images(files_list)

    return x_train, np.array(label)


def train_model(spectr_dir, train_csv, test_csv, model_type, use_cache = False, resume = False):

    print('Loading data... ')

    if not use_cache:
        x_train, y_train = load_data_from_spectrogram_dir(spectr_dir, ['h', 'p'], train_csv)
        x_test, y_test_ori = load_data_from_spectrogram_dir(spectr_dir, ['h', 'p'], test_csv)

        print('Saving data to cache...')
        np.save('train_data.npy', x_train)
        np.save('train_label.npy', y_train)
        np.save('val_data.npy', x_test)
        np.save('val_label.npy', y_test_ori)
    else:
        print('Loading data from cache...')
        x_train = np.load('train_data.npy')
        y_train = np.load('train_label.npy')
        x_test = np.load('val_data.npy')
        y_test_ori = np.load('val_label.npy')
        print(y_train.shape)

    label_count = np.bincount(y_train)
    print('Train label count: ', label_count)

    from sklearn.utils import class_weight
    y_train = list(y_train)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    print('sample weight per class', class_weights)

    print('-'*130)
    print ('Model train')
    print('-'*130)

    input_shape = (rows, cols, channels)

    from model import ResnetBuilder, CRNN, simple_CNN

    if model_type == "resnet18":
        model = ResnetBuilder.build_resnet_18(input_shape, num_classes)
    elif model_type == "resnet34":
        model = ResnetBuilder.build_resnet_18(input_shape, num_classes)
    elif model_type == "CRNN":
        model = CRNN(input_shape, num_classes)
    else:
        model = simple_CNN(input_shape, num_classes)

    optimizer = Adadelta(0.1, rho=0.7)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    x_train = x_train.reshape(x_train.shape[0], rows, cols, channels)
    x_test = x_test.reshape(x_test.shape[0], rows, cols, channels)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test_ori, num_classes)

    ### Load weights
    if resume:
        model.load_weights('./model/'+model_name)
    #
    checkpoint = ModelCheckpoint('./model/'+model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
    callbacks_list = [checkpoint, early_stop]

    print('epochs', epochs)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=1,
              shuffle=True,
              callbacks=callbacks_list,
              class_weight=class_weights
              )

    print('Report')
    y_prob = model.predict(x_test)
    y_pred = y_prob.argmax(axis=-1)
    print('test_y', np.bincount(y_test_ori))
    print(classification_report(y_test_ori, y_pred, target_names=CLASSES))

    model.save(model_name)


def evaluate():
    print('Loading data... ')
    x_train = np.load('val_data.npy')
    print('train shape', x_train.shape)
    y_label = np.load('val_label.npy')

    x_train = x_train.reshape(x_train.shape[0], rows, cols, channels)

    model = load_model('./model/' + model_name)

    print('Report')
    y_prob = model.predict(x_train, batch_size=32, verbose=1)
    y_pred = y_prob.argmax(axis=-1)
    print('test_y', np.bincount(y_label))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_label, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list(CLASSES), normalize=False,
                          title='Normalized confusion matrix')
    plt.savefig("cm.jpg")
    print(classification_report(y_label, y_pred, target_names=list(CLASSES)))

if __name__ == "__main__":
    if args.evaluate:
        evaluate()
    else:
        train_model(args.spectr_dir, args.train_csv, args.val_csv, args.model, args.use_cache, args.resume)

