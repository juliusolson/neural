#! /bin/bash

cd data
python3 dset.py

cd ..
go install all
nn-mnist