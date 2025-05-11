#!/bin/bash

wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.zip
for i in $(seq -f "%02g" 1 9); do
    wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z$i
done
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip -d ./datasets/
rm pix2code_datasets.zip
rm pix2code_datasets.z*
