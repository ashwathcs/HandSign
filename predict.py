#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:32:11 2019

@author: ashwathcs
"""

from keras.models import load_model

model = load_model('sign_detector.h5') 
image = to_detect
x = np.expand_dims(image, axis=0)
x = np.expand_dims(x, axis=0)
y = model.predict(x)
