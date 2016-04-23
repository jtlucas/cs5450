#!/usr/bin/python

from __future__ import print_function
import cPickle as pickle
import os

reviews = pickle.load(open("./data/train.p", "rb"))
