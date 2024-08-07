#!/bin/sh

curl -L https://www.kaggle.com/models/tensorflow/inception/tensorFlow1/tfgan-eval-inception/1\?tf-hub-format\=compressed \
    --output inception1.tar.gz

[ -d inception1/ ] && rm -r inception1/

mkdir inception1/ && tar -xzvf inception1.tar.gz -C inception1/

curl -L https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4?tf-hub-format=compressed \
    --output inception3.tar.gz 
    
[ -d inception3/ ] && rm -r inception3/

mkdir inception3/ && tar -xzvf inception3.tar.gz -C inception3/
