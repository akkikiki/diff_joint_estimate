#!/bin/bash

# Prepare directories to save figures
mkdir reports/figures/sentence_cefr/
mkdir reports/figures/word_cefr/

# Download the Cambridge exam dataset
wget -O data/raw/Readability_dataset.tar.gz https://s3-eu-west-1.amazonaws.com/ilexir-website-media/Readability_dataset.tar.gz
tar -C data/raw/ -zxvf data/raw/Readability_dataset.tar.gz

# visualize the sentences
python3 src/visualization/visualize_sentences.py

# visualize the words
python3 src/visualization/visualize_words.py
