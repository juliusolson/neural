#! /bin/bash

cd data
#python3 dset.py
DATADIR=$(pwd)

cd ..
go install all
#nn-mnist
echo $DATADIR
export DATADIR=$DATADIR
nn-mnist