#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pydub import AudioSegment
import os

dir='/Users/apple/Desktop/test'
for file in os.listdir(dir):
    ff=os.listdir(dir+'/'+file)
    for f in ff:
        s='mkdir {}/{}'.format(dir,f)
        os.system(s)
        fullname=os.path.join(ff,f)
        wav= AudioSegment.from_wav(fullname) 
        i=0
        while(wav):
            path=dir+'/'+f+'_'+i
            wav[i*1000:i*1000+1000].export(path, format="wav")
            i=i+1
        
        
        


