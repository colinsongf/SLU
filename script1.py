#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 21:43:36 2017

@author: shashank
"""
import re
import api_connect
# seqin=['find flights from miami to boston on saturday at 8 pm'].split(' ')
# seqout=['O O O B-fromloc.cityname O B-toloc.cityname O B-depart_date.day_name O B-depart_time.time B-depart_time.time_of_day'].split(' ')

def get_quotes(seqin, seqout) :
    tags=[]
    for line1,line2 in zip(seqin,seqout):
        words1=line1
        words2=line2
        line_tags={}
        assert len(words1)==len(words2)
        for word1,word2 in zip(words1,words2):
            if word2 not in line_tags:
                line_tags[word2]=[]
                line_tags[word2].append(word1)
            else:
                line_tags[word2].append(word1)
        tags.append(line_tags)

    for tag in tags:
        tag.pop('O',None)

    for tag in tags:
        temp_keys=tag.keys()
        temp_keys=sorted(temp_keys)
        for key in temp_keys:
            key1=re.sub(r'.*-','',key)
            key2=re.sub(r'(\.).*','',key1)
            if key2 in tag:
                tag[key2]+=tag.pop(key)
            else:
                tag[key2]=tag.pop(key)

    for tag in tags:
        for key in tag.keys():
            tag[key]=' '.join(tag[key])

    set_tags=set()
    for tag in tags:
        for key in tag.keys():
            set_tags.add(key)

    print tags[0]
    return api_connect.connect_api(tags[0])

