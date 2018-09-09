# Check full librosa spectrogram tutorial in the following IPython notebook:
# http://nbviewer.jupyter.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb

import librosa
import multiprocessing

import numpy as np
import scipy.misc
import os
import argparse
from tqdm import tqdm


def __get_output_filename(audio_fname):
    return os.path.join(spectr_dir, audio_fname + '.png')

def __extract_hpss_melspec(audio_fpath, audio_fname):
    """
    Extension of :func:`__extract_melspec`.
    Not used as it's about ten times slower, but
    if you have resources, try it out.

    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    y, sr = librosa.load(audio_fpath, sr=44100)

    # Harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    S_h = librosa.feature.melspectrogram(y_harmonic, sr=sr, n_mels=128)
    S_p = librosa.feature.melspectrogram(y_percussive, sr=sr, n_mels=128)

    log_S_h = librosa.power_to_db(S_h, ref=np.max)
    log_S_p = librosa.power_to_db(S_p, ref=np.max)

    spectr_fname_h = audio_fname + '_h'
    spectr_fname_p = audio_fname + '_p'

    scipy.misc.toimage(log_S_h).save(__get_output_filename(spectr_fname_h))
    scipy.misc.toimage(log_S_p).save(__get_output_filename(spectr_fname_p))


def __extract_melspec(audio_fpath, audio_fname):
    """
    Using librosa to calculate log mel spectrogram values
    and scipy.misc to draw and store them (in grayscale).

    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    # Load sound file
    y, sr = librosa.load(audio_fpath, sr=12000)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, hop_length=256, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S)

    spectr_fname = audio_fname + '_mel'

    # Draw log values matrix in grayscale
    scipy.misc.toimage(log_S).save(__get_output_filename(spectr_fname))


def __extract_cqt(audio_fpath, audio_fname):
    """
    Using librosa to compute the constant-Q transform of an audio signal
    and scipy.misc to draw and store them (in grayscale).

    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    # Load sound file
    y, sr = librosa.load(audio_fpath, sr=12000)

    fmin = librosa.midi_to_hz(36)
    hop_length = 512
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_C = librosa.amplitude_to_db(np.abs(C))

    spectr_fname = audio_fname + '_cqt'

    # Draw log values matrix in grayscale
    scipy.misc.toimage(log_C).save(__get_output_filename(spectr_fname))


def __extract_stft(audio_fpath, audio_fname):
    """
    Using librosa to calculate Short-time Fourier transform (STFT) values
    and scipy.misc to draw and store them (in grayscale).

    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    # Load sound file
    y, sr = librosa.load(audio_fpath, sr=12000)

    # And compute the spectrogram magnitude and phase
    S, phase = librosa.magphase(librosa.stft(y,  n_fft=1024))
    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S, ref=np.max)

    spectr_fname = audio_fname + '_stft'

    # Draw log values matrix in grayscale
    scipy.misc.toimage(log_S).save(__get_output_filename(spectr_fname))

def __extract_mfcc(audio_fpath, audio_fname):
    """
    Using librosa to calculate Mel-frequency cepstral coefficients (MFCCs) values
    and scipy.misc to draw and store them (in grayscale).

    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    # Load sound file
    y, sr = librosa.load(audio_fpath, sr=12000)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, hop_length=256, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)

    spectr_fname = audio_fname + '_mfcc'

    # Draw log values matrix in grayscale
    scipy.misc.toimage(mfcc).save(__get_output_filename(spectr_fname))


def __process_one_file(fpath):
    fname, _ = os.path.splitext(os.path.basename(fpath))
    # op_start_time = time.time()
    try:
        # print(__get_output_filename(fname + '_h'))
        if regenerate or not os.path.isfile(__get_output_filename(fname + '_h')):
            __extract_hpss_melspec(fpath, fname)
        # if os.path.isfile(__get_output_filename(fname + '_stft')):
        #     __extract_stft(fpath, fname)
        # if os.path.isfile(__get_output_filename(fname + '_mfcc')):
        #     __extract_mfcc(fpath, fname)
        # if os.path.isfile(__get_output_filename(fname + '_mel')):
        #     __extract_melspec(fpath, fname)
        # if os.path.isfile(__get_output_filename(fname + '_cqt')):
        #     __extract_cqt(fpath, fname)
    except Exception as e:
        print(e)
        return False
    return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data directory', default='/data')
    parser.add_argument('--output_dir', help='output directory', default='/tmp/spectr')
    parser.add_argument('--num_threads', type=int, help='number of processing threads', default=8)
    parser.add_argument('--regenerate', action='store_true', help='ovewrite existing spectrograms')
    args = parser.parse_args()

    # if args.regenerate:
    regenerate = args.regenerate
    music_dir = args.data_dir
    spectr_dir = args.output_dir

    if not os.path.exists(spectr_dir):
        os.mkdir(spectr_dir)

    files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) if f.endswith(".mp3")]
    nfiles = len(files)
    ok_cnt = 0
    fail_cnt = 0

    nb_workers = args.num_threads
    pool = multiprocessing.Pool(nb_workers)
    it = pool.imap_unordered(__process_one_file, files)

    for res in tqdm(it, total=len(files)):
        if res:
            ok_cnt += 1
        else:
            fail_cnt += 1

    pool.close()

    print('Generating spectrogram finished! Generated {}/{} images successfully'.format(ok_cnt, ok_cnt + fail_cnt))

