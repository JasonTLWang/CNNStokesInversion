# =========================================================================
#   (c) Copyright 2020
#   All rights reserved
#   Programs written by Hao Liu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

import csv
from astropy.io import fits
import numpy as np
import math
import os
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import time
import matplotlib.pyplot as plt
import matplotlib
import sys


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('turn off loggins is not supported')


def convert_fits_to_csv_train(time_list):
    print('Converting training fits to csv file...')
    csv_file_name = '../traincsv/train_full.csv'
    if os.path.exists(csv_file_name):
        os.remove(csv_file_name)
    with open(csv_file_name, 'a', encoding='UTF-8') as csv_file:
        for timepoint in time_list:
            w = csv.writer(csv_file)
            hdu = fits.open('../trainset/cals_' + timepoint + '.fts')
            hdu.verify('fix')
            data = hdu[0].data
            q = data[1, :, :, :]
            u = data[2, :, :, :]
            v = data[3, :, :, :]
            hdu = fits.open('../trainset/nirisinv_' + timepoint + '.fts')
            hdu.verify('fix')
            data = hdu[0].data
            b_total = data[0, :, :]
            inclination = data[1, :, :]
            azimuth = data[2, :, :]
            tmp_shape = b_total.shape
            b_total = b_total.reshape(-1, 1)
            inclination = inclination.reshape(-1, 1)
            azimuth = azimuth.reshape(-1, 1)
            b_total = np.clip(b_total, 0, 4999)
            inclination = np.clip(inclination, 0, math.pi - 1e-6)
            azimuth = np.clip(azimuth, 0, math.pi - 1e-6)
            b_total = b_total.reshape(tmp_shape)
            inclination = inclination.reshape(tmp_shape)
            azimuth = azimuth.reshape(tmp_shape)

            spectrum_height = q.shape[1]
            spectrum_width = q.shape[2]
            spectrum_length = q.shape[0]

            for i in range(spectrum_height):
                for j in range(spectrum_width):
                    q_spectrum = list()
                    u_spectrum = list()
                    v_spectrum = list()
                    for k in range(spectrum_length):
                        q_spectrum.append(int(q[k][i][j]))
                        u_spectrum.append(int(u[k][i][j]))
                        v_spectrum.append(int(v[k][i][j]))

                    if len(q_spectrum) < 60:
                        for k in range(int((60 - len(q_spectrum)) / 2)):
                            q_spectrum.insert(0, 0.)
                        for k in range(60 - len(q_spectrum)):
                            q_spectrum.append(0.)
                    if len(u_spectrum) < 60:
                        for k in range(int((60 - len(u_spectrum)) / 2)):
                            u_spectrum.insert(0, 0.)
                        for k in range(60 - len(u_spectrum)):
                            u_spectrum.append(0.)
                    if len(v_spectrum) < 60:
                        for k in range(int((60 - len(v_spectrum)) / 2)):
                            v_spectrum.insert(0, 0.)
                        for k in range(60 - len(v_spectrum)):
                            v_spectrum.append(0.)

                    labels = list()
                    labels.append(b_total[i][j])
                    labels.append(inclination[i][j])
                    labels.append(azimuth[i][j])

                    record = labels + q_spectrum + u_spectrum + v_spectrum
                    w.writerow(record)
    print('Done converting...')


def convert_fits_to_csv_test(testset_idx, time_list):
    print('Converting test fits to csv file...')
    csv_file_name = '../testset' + str(testset_idx) + '/testset' + str(testset_idx) + '.csv'
    if os.path.exists(csv_file_name):
        os.remove(csv_file_name)
    with open(csv_file_name, 'w', encoding='UTF-8') as csv_file:
        for timepoint in time_list:
            w = csv.writer(csv_file)
            hdu = fits.open('../testset' + str(testset_idx) + '/cals_' + timepoint + '.fts')
            hdu.verify('fix')
            data = hdu[0].data
            q = data[1, :, :, :]
            u = data[2, :, :, :]
            v = data[3, :, :, :]

            spectrum_height = q.shape[1]
            spectrum_width = q.shape[2]
            spectrum_length = q.shape[0]

            for i in range(spectrum_height):
                for j in range(spectrum_width):
                    q_spectrum = list()
                    u_spectrum = list()
                    v_spectrum = list()
                    for k in range(spectrum_length):
                        q_spectrum.append(int(q[k][i][j]))
                        u_spectrum.append(int(u[k][i][j]))
                        v_spectrum.append(int(v[k][i][j]))

                    if len(q_spectrum) < 60:
                        for k in range(int((60 - len(q_spectrum)) / 2)):
                            q_spectrum.insert(0, 0.)
                        for k in range(60 - len(q_spectrum)):
                            q_spectrum.append(0.)
                    if len(u_spectrum) < 60:
                        for k in range(int((60 - len(u_spectrum)) / 2)):
                            u_spectrum.insert(0, 0.)
                        for k in range(60 - len(u_spectrum)):
                            u_spectrum.append(0.)
                    if len(v_spectrum) < 60:
                        for k in range(int((60 - len(v_spectrum)) / 2)):
                            v_spectrum.insert(0, 0.)
                        for k in range(60 - len(v_spectrum)):
                            v_spectrum.append(0.)

                    record = q_spectrum + u_spectrum + v_spectrum
                    w.writerow(record)
    print('Done converting...')


def get_random_1000000():
    print('Randomly select one million data samples...')
    filename = '../traincsv/train_full.csv'
    newfilename = '../traincsv/train.csv'
    df = pd.read_csv(filename, header=None)
    df_values = df.values
    index = [i for i in range(len(df_values))]
    np.random.seed(123)
    idx_list = np.random.choice(index, 1000000)
    with open(newfilename, 'w', encoding='UTF-8') as newcsvfile:
        w = csv.writer(newcsvfile)
        for i in idx_list:
            w.writerow(df_values[i])
    os.remove(filename)
    print('Done...')


def normalization_test(testset_idx):
    print('Normalizing test data...')
    filename = '../testset' + str(testset_idx) + '/testset' + str(testset_idx) + '.csv'
    newfilename = '../testset' + str(testset_idx) + '/normalized_testset' + str(testset_idx) + '.csv'
    df = pd.read_csv(filename, header=None)
    df_values = df.values.astype(float)
    with open(newfilename, 'w', encoding='UTF-8') as new_csv_file:
        w = csv.writer(new_csv_file)
        for i in range(len(df_values)):
            df_values[i, :] = df_values[i, :] / 1000.
            w.writerow(df_values[i])
    print('Done Normalization...')


def normalization_train():
    print('Normalizing training data...')
    filename = '../traincsv/train.csv'
    newfilename = '../traincsv/normalized_train.csv'
    df = pd.read_csv(filename, header=None)
    df_values = df.values.astype(float)
    with open(newfilename, 'w', encoding='UTF-8') as new_csv_file:
        w = csv.writer(new_csv_file)
        for i in range(len(df_values)):
            df_values[i, 0] = df_values[i, 0] / 5000.
            df_values[i, 1] = df_values[i, 1] / math.pi
            df_values[i, 2] = df_values[i, 2] / math.pi
            df_values[i, 3:] = df_values[i, 3:] / 1000.
            w.writerow(df_values[i])
    print('Done Normalization...')


def load_train_data(filename):
    df = pd.read_csv(filename, header=None)
    df_values = df.values
    X_data = df_values[:, 3:]
    y = df_values[:, :3]
    X = X_data.reshape(X_data.shape[0], 60, 3)
    return X, y


def load_test_data(filename):
    df = pd.read_csv(filename, header=None)
    df_values = df.values
    X_data = df_values
    X = X_data.reshape(X_data.shape[0], 60, 3)
    return X


def create_model():
    input = Input(shape=(60, 3))
    conv0_1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input)
    conv0_2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv0_1)
    conv0 = MaxPool1D(pool_size=2)(conv0_2)
    conv1_1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv0)
    conv1_2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1_1)
    conv1 = MaxPool1D(pool_size=2)(conv1_2)
    conv2_1 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv1)
    conv2_2 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv2_1)
    cnn_out = Flatten()(conv2_2)
    layer1 = Dense(1024, activation='relu')(cnn_out)
    layer1 = Dropout(0.25)(layer1)
    layer2 = Dense(1024, activation='relu')(layer1)
    layer2 = Dropout(0.25)(layer2)
    output = Dense(3, activation='tanh')(layer2)
    model = Model(input, output)
    return model


def inverse(train_again, testset_idx):
    batch_size = 512
    epochs = 50
    train_file = '../traincsv/normalized_train.csv'
    test_file = '../testset' + str(testset_idx) + '/normalized_testset' + str(testset_idx) + '.csv'
    result_file = '../results' + str(testset_idx) + '/results' + str(testset_idx) + '.csv'
    model_file = './model.h5'

    if train_again:
        print('Loading training data...')
        X_train, y_train = load_train_data(train_file)
        print('Done loading...')

        model = create_model()
        model.compile(loss='mean_absolute_error', optimizer=Adam(decay=0.01), metrics=['mae'])
        print('Start training CNN model...')
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=True)
        print('Done training...')
        model.save(model_file)
    else:
        model = load_model(model_file)

    print('Loading testing data')
    X_test = load_test_data(test_file)
    print('Done loading...')
    print('Start inversion...')
    start = time.time()
    y_predict = model.predict(X_test)
    end = time.time()
    print('End inversion...')
    # print('Time:', np.round(end - start, 1), 's')

    print('Saving results...')
    with open(result_file, 'w', encoding='UTF-8') as result_csv:
        w = csv.writer(result_csv)
        for i in range(len(y_predict)):
            tmp = list()
            tmp.append(y_predict[i][0])
            tmp.append(y_predict[i][1])
            tmp.append(y_predict[i][2])
            w.writerow(tmp)
    print('Done saving...')


def plot(testset_idx, x_ticks, x_ticks_labels, y_ticks, y_ticks_labels):
    print('Generating figures...')
    file_name = '../results' + str(testset_idx) + '/results' + str(testset_idx) + '.csv'
    fontsize = 20
    titlesize = 24
    df = pd.read_csv(file_name, header=None)
    df_value = df.values
    df_value[:, [0]] = df_value[:, [0]] * 5000.
    df_value[:, [1, 2]] = df_value[:, [1, 2]] * math.pi
    predicted_bx = np.multiply(np.multiply(df_value[:, 0], np.sin(df_value[:, 1])), np.cos(df_value[:, 2]))
    predicted_by = np.multiply(np.multiply(df_value[:, 0], np.sin(df_value[:, 1])), np.sin(df_value[:, 2]))
    predicted_bz = np.multiply(df_value[:, 0], np.cos(df_value[:, 1]))
    df_value[:, [1, 2]] = df_value[:, [1, 2]] / math.pi * 180.
    height = 720
    width = 720
    predicted_btotal_img = df_value[:, 0].reshape(height, width)
    predicted_inclination_img = df_value[:, 1].reshape(height, width)
    predicted_azimuth_img = df_value[:, 2].reshape(height, width)
    predicted_bx_img = predicted_bx.reshape(height, width)
    predicted_by_img = predicted_by.reshape(height, width)
    predicted_bz_img = predicted_bz.reshape(height, width)
    ref_btotal_img = np.ones((height, width)) * 4000
    ref_btotal_img[0][0] = 0
    ref_angle_img = np.ones((height, width)) * 180
    ref_angle_img[0][0] = -5
    ref_bx_img = np.ones((height, width)) * 3000
    ref_bx_img[0][0] = -3500
    ref_by_img = np.ones((height, width)) * 3000
    ref_by_img[0][0] = -3500
    ref_bz_img = np.ones((height, width)) * 3000
    ref_bz_img[0][0] = -3500
    my_map = 'RdBu'
    img_width = 10
    img_height = 10
    # B_total
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(img_width, img_height))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=4000)
    im = ax.imshow(ref_btotal_img, cmap=my_map, norm=norm)
    ax.imshow(predicted_btotal_img, cmap=my_map, norm=norm)
    ax.set_ylabel('N-S Direction (arcsecond)', fontsize=fontsize)
    ax.set_xlabel('E-W Direction (arcsecond)', fontsize=fontsize)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax.set_title('CNN, $B_{total}$', x=0.85, y=0, fontsize=titlesize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.15, hspace=0.15)
    cax = plt.axes([0.92, 0.2, 0.02, 0.6])  # bottom left width height
    matplotlib.rcParams.update({'font.size': fontsize})
    cb = fig.colorbar(im, cax=cax, norm=norm)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('(Gauss)', fontsize=fontsize)
    plt.savefig('../results' + str(testset_idx) + '/Btotal.png', bbox_inches='tight')

    # inclination
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(img_width, img_height))
    norm = matplotlib.colors.Normalize(vmin=-5, vmax=180)
    im = ax.imshow(ref_angle_img, cmap=my_map, norm=norm)
    ax.imshow(predicted_inclination_img, cmap=my_map, norm=norm)
    ax.set_ylabel('N-S Direction (arcsecond)', fontsize=fontsize)
    ax.set_xlabel('E-W Direction (arcsecond)', fontsize=fontsize)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax.set_title('CNN, Inclination', x=0.76, y=0, fontsize=titlesize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.15, hspace=0.15)
    cax = plt.axes([0.925, 0.2, 0.02, 0.6])  # bottom left width height
    matplotlib.rcParams.update({'font.size': fontsize})
    cb = fig.colorbar(im, cax=cax, norm=norm)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('(Degree)', fontsize=fontsize)
    plt.savefig('../results' + str(testset_idx) + '/Inclination.png', bbox_inches='tight')

    # azimuth
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(img_width, img_height))
    norm = matplotlib.colors.Normalize(vmin=-5, vmax=180)
    im = ax.imshow(ref_angle_img, cmap=my_map, norm=norm)
    ax.imshow(predicted_azimuth_img, cmap=my_map, norm=norm)
    ax.set_ylabel('N-S Direction (arcsecond)', fontsize=fontsize)
    ax.set_xlabel('E-W Direction (arcsecond)', fontsize=fontsize)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax.set_title('CNN, Azimuth', x=0.8, y=0, fontsize=titlesize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.15, hspace=0.15)
    cax = plt.axes([0.925, 0.2, 0.02, 0.6])  # bottom left width height
    matplotlib.rcParams.update({'font.size': fontsize})
    cb = fig.colorbar(im, cax=cax, norm=norm)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('(Degree)', fontsize=fontsize)
    plt.savefig('../results' + str(testset_idx) + '/Azimuth.png', bbox_inches='tight')

    # Bx
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(img_width, img_height))
    norm = matplotlib.colors.Normalize(vmin=-3500, vmax=3000)
    im = ax.imshow(ref_bx_img, cmap=my_map, norm=norm)
    ax.imshow(predicted_bx_img, cmap=my_map, norm=norm)
    ax.set_ylabel('N-S Direction (arcsecond)', fontsize=fontsize)
    ax.set_xlabel('E-W Direction (arcsecond)', fontsize=fontsize)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax.set_title('CNN, $B_x$', x=0.85, y=0, fontsize=titlesize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.15, hspace=0.15)
    cax = plt.axes([0.92, 0.2, 0.02, 0.6])  # bottom left width height
    matplotlib.rcParams.update({'font.size': fontsize})
    cb = fig.colorbar(im, cax=cax, norm=norm)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('(Gauss)', fontsize=fontsize)
    plt.savefig('../results' + str(testset_idx) + '/Bx.png', bbox_inches='tight')

    # By
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(img_width, img_height))
    norm = matplotlib.colors.Normalize(vmin=-3500, vmax=3000)
    im = ax.imshow(ref_by_img, cmap=my_map, norm=norm)
    ax.imshow(predicted_by_img, cmap=my_map, norm=norm)
    ax.set_ylabel('N-S Direction (arcsecond)', fontsize=fontsize)
    ax.set_xlabel('E-W Direction (arcsecond)', fontsize=fontsize)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax.set_title('CNN, $B_y$', x=0.85, y=0, fontsize=titlesize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.15, hspace=0.15)
    cax = plt.axes([0.92, 0.2, 0.02, 0.6])  # bottom left width height
    matplotlib.rcParams.update({'font.size': fontsize})
    cb = fig.colorbar(im, cax=cax, norm=norm)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('(Gauss)', fontsize=fontsize)
    plt.savefig('../results' + str(testset_idx) + '/By.png', bbox_inches='tight')

    # Bz
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(img_width, img_height))
    norm = matplotlib.colors.Normalize(vmin=-3500, vmax=3000)
    im = ax.imshow(ref_bz_img, cmap=my_map, norm=norm)
    ax.imshow(predicted_bz_img, cmap=my_map, norm=norm)
    ax.set_ylabel('N-S Direction (arcsecond)', fontsize=fontsize)
    ax.set_xlabel('E-W Direction (arcsecond)', fontsize=fontsize)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax.set_title('CNN, $B_z$', x=0.85, y=0, fontsize=titlesize)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.15, hspace=0.15)
    cax = plt.axes([0.92, 0.2, 0.02, 0.6])  # bottom left width height
    matplotlib.rcParams.update({'font.size': fontsize})
    cb = fig.colorbar(im, cax=cax, norm=norm)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('(Gauss)', fontsize=fontsize)
    plt.savefig('../results' + str(testset_idx) + '/Bz.png', bbox_inches='tight')
    print('Done...')


if __name__ == '__main__':
    testset_idx = int(sys.argv[1])
    train_again = int(sys.argv[2])
    convert_date_again = int(sys.argv[3])
    if convert_date_again == 1:
        time_list = ['150622_173300', '150622_173400']
        convert_fits_to_csv_train(time_list)
        get_random_1000000()
        normalization_train()
        if testset_idx == 1:
            convert_fits_to_csv_test(testset_idx, ['150625_200000'])
            normalization_test(testset_idx)
        elif testset_idx == 2:
            convert_fits_to_csv_test(testset_idx, ['170713_183500'])
            normalization_test(testset_idx)
        elif testset_idx == 3:
            convert_fits_to_csv_test(testset_idx, ['170906_191800'])
            normalization_test(testset_idx)

    inverse(train_again, testset_idx)
    if testset_idx == 1:
        plot(testset_idx, [94, 212, 330, 449, 567, 685],
                        ['650', '660', '670', '680', '690', '700'],
                        [105, 221, 337, 453, 569, 685],
                        ['-170', '-180', '-190', '-200', '-210', '-220'])
    elif testset_idx == 2:
        plot(testset_idx, [24, 142, 260, 378, 496, 614],
                        ['-470', '-460', '-450', '-440', '-430', '-420'],
                        [71, 189, 307, 425, 543, 661],
                        ['200', '190', '180', '170', '160', '150'])
    elif testset_idx == 3:
        plot(testset_idx, [0, 118, 236, 354, 472, 590, 708],
                        ['560', '570', '580', '590', '600', '610', '620'],
                        [71, 189, 307, 425, 543, 661],
                        ['-210', '-220', '-230', '-240', '-250', '-260'])
