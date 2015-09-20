#!/usr/bin/python

import sys
import re
import os.path
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def cross(list1, list2):
    if( len(list1)==0 and len(list2)==0 ):
        return []
    elif( len(list1)==0 ):
        return list2
    elif( len(list2)==0 ):
        return list1

    result = []
    try:
        for i in xrange( len(list1) ):
            for j in xrange( len(list2) ):
                head = list1[i]
                if( type(head) is not list ):
                    head = [head]
                tail = list2[j]
                if( type(tail) is not list ):
                    tail = [tail]

                result.append( head + tail )
    except:
        print("some thing wrong")
    return result

def buildPhrase(word_set):
    """
    N x M list
    word_set = [
                            [ sim1_to_word1, sim2_to_word1, ... simM_to_word1],
                            [ sim1_to_word2, sim2_to_word2, ... simM_to_word2],
                            ...
                            [ sim1_to_wordN, sim2_to_wordN, ... simM_to_wordN]
                        ]
    (M^N) x N list
    result = [  
                        [ sim1_to_word1, sim1_to_word2, ... sim1_to_wordN],
                        [ sim1_to_word1, sim1_to_word2, ... sim2_to_wordN],
                        ...
                        [ sim1_to_word1, simM_to_word2, ... simM_to_wordN],
                        [ sim2_to_word1, sim1_to_word2, ... sim1_to_wordN],
                        [ sim2_to_word1, sim1_to_word2, ... sim2_to_wordN],
                        ...
                        [ simM_to_word1, simM_to_word2, ... simM_to_wordN]
                    ]
    usage:
    word_set = [
                            [ 'sim1_to_word1', 'sim2_to_word1'],
                            [ 'sim1_to_word2', 'sim2_to_word2'],
                            [ 'sim1_to_word3', 'sim2_to_word3']
                        ]
    combined_phrase = buildPhrase(word_set)
    print(combined_phrase)
    """
    result = []
    for words in word_set:
        result = cross( result, words )
    return result

if __name__=="__main__":

    last = sys.argv[1]
    build_model = sys.argv[2]

    #pathes
    training_dir = 'training_labels/'
    seg_manual = training_dir + "chinese.label.manual."
    extend_path = training_dir + "chinese.label.extend."
    model_path = 'word2vec.model'
    raw_text_path = "Chinese.txt"
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
    for i,phrase in enumerate(qualified_labels):
        try:
            word_set = []
            segments = [ unicode(x) for x in phrase.split(' ')]
            if(len(segments) > 3):
                continue
            topn = 14 - len(segments)*2
            for j,word in enumerate(segments):
                candidate = model.most_similar_cosmul(positive=[word],topn=topn)
                trimed_list = [word]    #add word itself
                for unigram in candidate:
                    #similarity filter
                    if(float(unigram[1])<0.7 ):
                        continue
                    #remove dulplicated word
                    trimed = re.search(ur"[\w].*[\w]",unigram[0]).group()
                    if( trimed.lower()!=word.lower() and trimed not in trimed_list):
                        trimed_list.append( trimed )
                if( len(trimed_list)!=0):
                    word_set.append( trimed_list )
                #find syn's similar, if they are too similar to syn_list, then it is syntagmatic word of segment word
            examee_list = buildPhrase(word_set)
            #print("similarity from " + phrase + " : ")
            for j,examee in enumerate(examee_list):
                if(len(examee) != len(set(examee)) ):
                    continue
                try:
                    sim_from_phrase = model.n_similarity( segments, examee )
                    if(sim_from_phrase == True or sim_from_phrase > 0.9999 ):
                        continue    #same phrase
                    examee_name = unicode(u" ".join(examee))
                    #print( " ".join(examee) + " : " +str(sim_from_phrase) )
                    #sim_map[unicode(u" ".join(examee))] = sim_from_phrase
                    if(examee_name not in sim_map):
                        sim_map[examee_name] = sim_from_phrase
                    elif(examee_name in sim_map and sim_from_phrase > sim_map[examee_name]):
                        sim_map[examee_name] = sim_from_phrase
                except:
                    print(" Something wrong with " + str(segments) + "to" + str(examee))
        
        except:
            print( "Something wrong happened when building candidate for [" + phrase +"]")
            continue

    print( "sim_map count : " + str( len(sim_map) ) )

    #sort sim_map by similarity
    map_to_list = sorted( sim_map, key=sim_map.__getitem__, reverse=True )
    with open("sim_map" ,"w") as map_file:
        for proto in map_to_list:
            try:
                map_file.write(proto + "\t" + unicode(sim_map[proto]) + "\n")
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