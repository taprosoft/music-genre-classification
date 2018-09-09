cd data && unzip train.zip
cd ..
mkdir -p spectr/train
python src/preprocess.py --data_dir='data/train' --output_dir='spectr/train'
python src/util.py --input='./data/train.csv' --train_ratio=0.9