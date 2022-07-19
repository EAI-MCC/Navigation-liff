#!/usr/bin/env bash

cd data
rm -f *.h5
wget http://vision.stanford.edu/yukezhu/thor_v1_scene_dumps_all.zip
unzip thor_v1_scene_dumps_all.zip
rm thor_v1_scene_dumps_all.zip
cd ..
