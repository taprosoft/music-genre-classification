import argparse

rows = 128     ### sliced spectrogram height
cols = 1024    ### sliced spectrogram width
channels = 2

max_width = 10337 ### maximum raw spectrogram width
split_count = 10  ### number of slices per song

epochs = 100
batch_size = 64
spectrogram_features = ['h', 'p']   ### Percussive & harmonic component spectrogram

CLASSES = ['Cai Luong', 'Cach Mang', 'Dan Ca - Que Huong', 'Dance', 'Khong Loi',
               'Thieu Nhi', 'Trinh', 'Tru Tinh', 'Rap Viet', 'Rock Viet']
num_classes = len(CLASSES)

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str, default="data/train_set.csv", help='path to train set csv')
parser.add_argument("--val_csv", type=str, default="data/val_set.csv", help='path to validation set csv')
parser.add_argument("--spectr_dir", type=str, default="data/spectr/train", help='path to train spectrogram images')

parser.add_argument("--model", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "CRNN", "simpleCNN"], help='model type')
parser.add_argument("--checkpoint", type=str, default='model_cnn_genre.h5', help='path to checkpoint')

parser.add_argument('--evaluate', action='store_true', help='evaluate trained model with validation data')
parser.add_argument('--use_cache', action='store_true', help='use cached .npy data from disk')
parser.add_argument('--resume', action='store_true', help='resume training from latest checkpoint')
args = parser.parse_args()


model_name = args.checkpoint   ### Saved model name