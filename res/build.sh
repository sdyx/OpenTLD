#!/bin/bash

DIR=~/repos/sdyx-tld/build
PWD=$( pwd )
cd ${DIR}
rm -rf {$DIR}/*
cmake ${DIR}/..
make
cp ${DIR}/../res/*.xml ${DIR}/
cp ${DIR}/../res/*.dat ${DIR}/
cp ${DIR}/../res/*.fel ${DIR}/
