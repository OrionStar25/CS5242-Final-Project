#!/bin/zsh

python3 mlp.py --train --data_path ../data/train_imdb.csv --model_path model.pt && \
python3 mlp.py --test --data_path ../data/test_imdb.csv --model_path model.pt --output_path out.txt
