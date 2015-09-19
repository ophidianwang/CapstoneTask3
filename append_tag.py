#!/usr/bin/python

import sys

last = sys.argv[1]

extend_dir = 'training_labels/'
source_path = extend_dir + "chinese.label.source."
extend_path = extend_dir + "chinese.label.extend."

for i in xrange( int(last) + 1 ):
    # from previous Word2Vec result
    current_file_name = source_path + str(i)
    print("try read file " + current_file_name )
    try:
        with open( current_file_name, "r") as labels_file, open( extend_path + str(i), "w" ) as output_file:
            for line in labels_file:
                output_file.write( line.strip() + "\t1\n" )
    except:
        print("read file " + current_file_name +" failed")