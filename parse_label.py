#!/usr/bin/python

import sys
import string
import shutil

last = sys.argv[1]

#dir pathes
training_dir = 'training_labels/'
seg_data_dir = '../SegPhrase/data/'
#manual judged data
source = training_dir + 'chinese.label.manual.'
#training log
label_quality = training_dir + 'chinese_labels_quality_' + str(last) + '.txt'
label_all = training_dir + 'chinese_labels_all_' + str(last) + '.txt'
#output for SegPhrase training
seg_knowlege_quality = seg_data_dir + "chinese_labels_quality.txt"
seg_knowlege_all = seg_data_dir + "chinese_labels_all.txt"

labels = {} # dict, key is label name, value is quality or not
for cursor in range( 0, int(last)+1 ):
    print("open file " + source+str(cursor) )
    with open(source+str(cursor), 'r') as inFile:
        for line in inFile:
            line = line.strip()
            tmp = line.split("\t")
            if( int(tmp[1]) == 1):
                labels[ tmp[0] ] = True
            else:
                if( not (tmp[0] in labels) ):
                    labels[ tmp[0] ] = False
    print( "labels size: " + str( len(labels) ) )

with open(label_quality,'w') as file_quality, open(label_all, 'w') as file_all:
    for key in labels:
        file_all.write(key+"\n")
        if( labels[key] ):
        	file_quality.write(key+"\n")

shutil.copyfile( label_quality , seg_knowlege_quality)
shutil.copyfile( label_all , seg_knowlege_all)

print("done")