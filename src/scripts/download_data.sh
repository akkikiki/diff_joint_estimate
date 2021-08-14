# Download GloVe
mkdir -p data/raw/glove/
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip -d data/raw/glove/ glove.6B.zip
rm glove.6B.zip


# Download Cambridge Readability Dataset
wget https://s3-eu-west-1.amazonaws.com/ilexir-website-media/Readability_dataset.tar.gz
tar xvf Readability_dataset.tar.gz -C data/raw/

