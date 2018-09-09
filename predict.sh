python src/preprocess.py --data_dir='data/private' --output_dir='spectr/private'
python src/predict.py --spectr_dir='spectr/private' --song_dir='data/private' --output_file='submission.csv'