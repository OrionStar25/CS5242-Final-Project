#!/bin/zsh

python3 cnn.py --train --data_path ../data/cleaned_imdb.csv --model_path model.pt && \
python3 cnn.py --test --data_path ../data/cleaned_imdb.csv --model_path model.pt --output_path out.txt
