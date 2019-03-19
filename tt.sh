#!/bin/sh

rm -rf img/
echo "`date +'%Y/%m/%d %T'` - Start"
python train_mnists.py #2> /dev/null
echo "`date +'%Y/%m/%d %T'` - End"
