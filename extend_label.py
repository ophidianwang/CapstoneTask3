#!/usr/bin/python

import sys
import shutil
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

    #pathes
    extend_dir = 'extend_labels/'
    labels_path = extend_dir + "chinese.label.manual."
    auto_path = extend_dir + "chinese.label.extend."
    raw_text_path = "../SegPhrase/data/Chinese.txt"
    patterns_path = "../SegPhrase/results/patterns.csv"

    #copy the orignial file
    shutil.copyfile( labels_path + str(last) , labels_path + str( int(last)+1 ))

    #load quality labels
    quality_labels = []
    with open(labels_path + str(last), "r") as labels_file:
        for line in labels_file:
            quality_labels.append( line.strip() )
                
    print( "quality_labels count : " + str( len(quality_labels) ) )

    #load pattern built by SegPhrase
    all_patterns = [] 
    with open(patterns_path,'r') as patterns_file:
        for line in (patterns_file):
            tmp = line.strip().split(',')
            all_patterns.append( tmp[0] )

    #load raw_text
    sentences = LineSentence(raw_text_path)

    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=2)
    model.init_sims(replace=True)

    proto_phrases = []
    for phrase in quality_labels:
        try:
            word_set = []
            segments = phrase.split(' ')
            topn = 14 - len(segments)*2
            for word in segments:
                candidate = model.most_similar(positive=[word],topn=topn)
                word_set.append( dict(candidate).keys() )
            proto_phrases += buildPhrase(word_set)
        except:
            print( "Something wrong happened when building candidate for [" + phrase +"]")
            continue

    print( "proto_phrases count : " + str( len(proto_phrases) ) )

    possible_phrases = []
    #exam if the possible exist in pattern.csv
    for i,phrase in enumerate(proto_phrases):
        target = ' '.join(phrase)
        if( target in all_patterns ):
            if( target not in possible_phrases and target not in quality_labels):
                print("#" + str(i) + " hit : " + target)
                possible_phrases.append(target)

    print( "possible_phrases count : " + str( len(possible_phrases) ) )

    #append extend labels
    with open(labels_path + str( int(last)+1 ) ,'a') as extend_file:
        for phrase in possible_phrases:
            extend_file.write( phrase + "\t1\n" )

    shutil.copyfile( labels_path + str( int(last)+1 ) , auto_path + str( int(last)+1 ) )