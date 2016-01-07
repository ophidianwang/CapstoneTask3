#!/usr/bin/python

import sys
import string
import shutil

now = sys.argv[1]

# dir pathes
training_dir = 'training_labels/'
seg_dir = '../SegPhrase/data/'
source = seg_dir + 'chinese.label.auto'
file_auto = training_dir + 'chinese.label.auto.' + str(now)
file_manual = training_dir + 'chinese.label.manual.' + str(now)

shutil.copyfile( source , file_auto)
shutil.copyfile( source , file_manual)