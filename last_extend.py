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
        model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=2)
        model.save(model_path)
        model.init_sims(replace=True)

    proto_phrases = []
    for phrase in qualified_labels:
        try:
            word_set = []
            segments = phrase.split(' ')
            topn = 14 - len(segments)*2
            for word in segments:
                candidate = model.most_similar(positive=[word],topn=topn)
                #print(word + " similiar to :")
                trimed_list = []
                for unigram in candidate:
                    #similarity filter
                    if(float(unigram[1])<0.75 ):
                        continue
                    #remove dulplicated word
                    trimed = re.search(ur"[\w].*[\w]",unigram[0]).group()
                    if(len(trimed)<2):
                        continue
                    if( trimed.lower()!=word.lower() and trimed not in trimed_list):
                        trimed_list.append( trimed )
                #print(trimed_list)
                if( len(trimed_list)!=0):
                    word_set.append( trimed_list )
            proto_phrases += buildPhrase(word_set)
        
        except:
            print( "Something wrong happened when building candidate for [" + phrase +"]")
            continue

    print( "proto_phrases count : " + str( len(proto_phrases) ) )

    with open(result_path ,"w") as result_file:
        result_file.write("Chinese\n")
        cursor = 0
        for phrase in qualified_labels:
            result_file.write( phrase + "\n" )
            cursor +=1
        for proto in proto_phrases:
            phrase = " ".join(proto)
            if( phrase not in qualified_labels):
                #print(phrase)
                try:
                    result_file.write( phrase + "\n" )
                except:
                    continue
                cursor +=1
            if(cursor > 9999):
                print("reach 10000, done")
                break
        else:
            print("less than 10000, done")