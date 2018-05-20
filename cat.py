#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

dir='/home/wcq/derja0805'
files = os.listdir(dir)
for file in files:
    k=1
    if(os.path.isdir(file) and len(os.listdir(file))<7):
        ff=os.listdir('/home/wcq/derja0805/'+file)
        for f in ff:
            if(k):
                s='cp {}/{} enroll/{}_{}'.format(file,f,file,f)
                os.system(s)
                k=k-1
            else:
                s='cp {}/{} test/{}_{}'.format(file,f,file,f)
                os.system(s)

    '''
    for f in ff:
        if(k):
            s='cp {}/{} enroll'.format(file,f)
            os.system(s)
            k=k-1
        if(k==0):
            s='cp {}/{} test'.format(file,f)
            os.system(s)
    '''