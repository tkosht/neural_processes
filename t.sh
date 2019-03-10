#!/bin/sh

rm -rf img/
echo "`date +'%Y/%m/%d %T'` - Start"
python train.py
echo "`date +'%Y/%m/%d %T'` - End"
