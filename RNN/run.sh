#!/bin/zsh

python3 rnn.py --train --data_path ../data/cleaned_imdb.csv --model_path model.pt && \
python3 rnn.py --test --data_path ../data/cleaned_imdb.csv --model_path model.pt --output_path out.txt
