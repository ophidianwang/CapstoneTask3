#!/usr/bin/python

import sys

if __name__=="__main__":

    last = sys.argv[1]

    #pathes
    training_dir = 'training_labels/'
    seg_manual = training_dir + "chinese.label.manual."
    extend_path = training_dir + "chinese.label.extend."
    current_file_name = ""
    result_path = "result32.txt"


    #load quality labels
    qualified_labels = []
    for i in xrange( int(last) + 1 ):
        # from previous Word2Vec result
        current_file_name = extend_path + str(i)
        print("try read file " + current_file_name )
        try:
            with open( current_file_name, "r") as labels_file:
                for line in labels_file:
                    line = line.strip()
                    tmp = line.split("\t")
                    if( int(tmp[1]) == 1 and tmp[0] not in qualified_labels):
                        qualified_labels.append( tmp[0] )
        except:
            print("read file " + current_file_name +" failed")

        # from previous SegPhrase result
        current_file_name = seg_manual+str(i)
        print("try read file " + current_file_name )
        try:
            with open( current_file_name, 'r') as seg_file:
                for line in seg_file:
                    line = line.strip()
                    tmp = line.split("\t")
                    if( int(tmp[1]) == 1 and tmp[0] not in qualified_labels):
                        qualified_labels.append( tmp[0] )
        except:
            print("read file " + current_file_name +" failed")
                
    print( "qualified_labels count : " + str( len(qualified_labels) ) )

    #proto phrase
    proto_phrase = []
    with open("sim_map","r") as sim_file:
        for line in sim_file:
            tmp = line.strip().split("\t")
            if(float(tmp[1]) < 0.9):
                break
            proto_phrase.append(unicode(tmp[0]))

    with open(result_path ,"w") as result_file:
        result_file.write("Chinese\n")
        cursor = 0
        for phrase in qualified_labels:
            result_file.write( phrase + "\n" )
            cursor +=1
        for proto in proto_phrase:
            if( proto not in qualified_labels):
                try:
                    result_file.write( proto + "\n" )
                except:
                    continue
                cursor +=1
            if(cursor > 9999):
                print("reach 10000, done")
                break
        else:
            print("less than 10000, done")
   
