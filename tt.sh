#!/bin/sh

rm -rf img/
echo "`date +'%Y/%m/%d %T'` - Start"
python train_mnists.py \
    --batch-size=30 --lr=0.001 \
    --epoch=1000 --log-interval=100 --fix-iter=100 \
    --dataset=fashion
echo "`date +'%Y/%m/%d %T'` - End"
