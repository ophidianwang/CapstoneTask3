#!/usr/bin/python

import sys
import re
import os.path
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__=="__main__":

    last = sys.argv[1]
    build_model = sys.argv[2]

    #pathes
    training_dir = 'training_labels/'
    seg_manual = training_dir + "chinese.label.manual."
    extend_path = training_dir + "chinese.label.extend."
    model_path = 'word2vec.model'
    raw_text_path = "Chinese.txt"
    patterns_path = "../SegPhrase/results/patterns.csv"
    current_file_name = ""
    result_path = "from_pattern_32.txt"

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

    #load pattern built by SegPhrase
    all_patterns = [] 
    with open(patterns_path,'r') as patterns_file:
        for line in (patterns_file):
            tmp = line.strip().split(',')
            all_patterns.append( tmp[0] )
    print( "all_patterns count : " + str( len(all_patterns) ) )
    
    #load model if exist
    if( os.path.isfile(model_path) and int(build_model)!= 1):
        print( "load existing model" )
        model = Word2Vec.load( model_path )
    else:
        print( "build model" )
        #load raw_text
        sentences = LineSentence(raw_text_path)
        model = Word2Vec(sentences, size=200, window=3, min_count=3, workers=2)
        model.save(model_path)
        model.init_sims(replace=True)

    sim_map = {}
    sim_to_label = {}
    for i,pattern in  enumerate(all_patterns):
        print("#"+str(i)+" : "+pattern)
        #if(len(sim_map) > 20 ):
            #break
        try:
            unigram_in_pattern = []
            temp = pattern.split(' ')
            for unigram in temp:
                trimed = re.search(ur"[\w].*[\w]",unigram).group().lower()
                if( trimed not in unigram_in_pattern):
                    unigram_in_pattern.append( trimed )
            #print(unigram_in_pattern)

            for j, phrase in enumerate(qualified_labels):
                try:
                    segments = [ unicode(x) for x in phrase.split(' ')]
                    sim_from_phrase = model.n_similarity( segments, unigram_in_pattern )
                    if(pattern not in sim_map):
                        sim_map[pattern] = sim_from_phrase
                        sim_to_label[pattern] = phrase
                    elif(pattern in sim_map and sim_from_phrase > sim_map[pattern]):
                        sim_map[pattern] = sim_from_phrase
                        sim_to_label[pattern] = phrase
                except:
                    continue
                    #print( "Something wrong happened when measure similarity from [" + pattern +"] to [" + phrase + "]")
        except:
            print( "Something wrong happened when spliting [" + pattern +"]")

    #sort sim_map by similarity
    map_to_list = sorted( sim_map, key=sim_map.__getitem__, reverse=True )
    with open("sim_map" ,"w") as map_file:
        for proto in map_to_list:
            try:
                map_file.write(proto + "\t" + unicode(sim_map[proto]) + "\t" + unicode(sim_to_label[proto]) + "\n")
            except:
                #print(" Something wrong with " + unicode(proto) )
                continue

    with open(result_path ,"w") as result_file:
        result_file.write("Chinese\n")
        cursor = 0
        for phrase in qualified_labels:
            result_file.write( phrase + "\n" )
            cursor +=1
        for proto in map_to_list:
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