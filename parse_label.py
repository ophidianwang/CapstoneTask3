#!/usr/bin/python

import sys
import string
import shutil

last = sys.argv[1]

#dir pathes
training_dir = 'training_labels/'
seg_data_dir = '../SegPhrase/data/'
#manual judged data
seg_manual = training_dir + 'chinese.label.manual.'
extend_manual =  training_dir + 'chinese.label.extend.'
#output file
label_quality = training_dir + 'chinese_labels_quality_' + str(last) + '.txt'
label_all = training_dir + 'chinese_labels_all_' + str(last) + '.txt'
#label file in SegPhrase
seg_knowlege_quality = seg_data_dir + "chinese_labels_quality.txt"
seg_knowlege_all = seg_data_dir + "chinese_labels_all.txt"

qualified_labels = []
all_labels = []

current_file_name = ""
with open(label_quality,'w') as file_quality, open(label_all, 'w') as file_all:
    for cursor in range( 0, int(last)+1 ):
        
        # from modified SegPhrase
        current_file_name = seg_manual+str(cursor)
        print("try read file " + current_file_name )
        try:
            with open(seg_manual+str(cursor), 'r') as in_file:
            
                for line in in_file:
                    line = line.strip()
                    tmp = line.split("\t")
                    if( int(tmp[1]) == 1 and tmp[0] not in qualified_labels):
                        qualified_labels.append( tmp[0] )
                        file_quality.write( tmp[0] + "\n" )
                    
                    if( tmp[0] not in all_labels):
                        all_labels.append( tmp[0] )
                        file_all.write( tmp[0] + "\n" )
        except:
            print("read file " + current_file_name +" failed")

        # from modified Word2Vec extend
        current_file_name = extend_manual+str(cursor)
        print("try read file " + current_file_name )
        try:
            with open( current_file_name, 'r') as in_file:
                for line in in_file:
                    line = line.strip()
                    tmp = line.split("\t")
                    if( int(tmp[1]) == 1 and tmp[0] not in qualified_labels):
                        qualified_labels.append( tmp[0] )
                        file_quality.write( tmp[0] + "\n" )
                    
                    if( tmp[0] not in all_labels):
                        all_labels.append( tmp[0] )
                        file_all.write( tmp[0] + "\n" )
        except:
            print("read file " + current_file_name +" failed")

        print( "qualified labels size: " + str( len(qualified_labels) ) )
        print( "all labels size: " + str( len(all_labels) ) )

shutil.copyfile( label_quality , seg_knowlege_quality)
shutil.copyfile( label_all , seg_knowlege_all)

print("done")