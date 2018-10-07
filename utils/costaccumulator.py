'''
author:         szechi
date created:   2 Oct 2018
description:    class to accumulate costs
'''

import os
import sys

class costAccumulator:
    __costdict = None
    __divisorcount = None
    __divided = None

    def __init__(self, costnames):
        self.__costdict = {str(k):0.0 for k in costnames}
        self.__newcosts = {str(k):0.0 for k in costnames}
        self.__divisorcount = 0
        self.__divided = False

    def getCost(self, costname):
        return self.__costdict[costname]

    def getAll(self):
        return self.__costdict

    def addCost(self, costname, costadd):
        self.__newcosts[costname] = costadd
        self.__costdict[costname] += costadd

    def resetCost(self):
        self.__costdict = dict.fromkeys(self.__costdict, 0.0)
        self.__newcosts = dict.fromkeys(self.__newcosts, 0.0)
        self.__divisorcount = 0
        self.__divided = False

    def addDivisorCount(self, num=1):
        if self.__divided:
            print "Warning: Quotient already taken"
        self.__divisorcount += num

    def makeMean(self):
        if self.__divided:
            print "Warning: Quotient already taken"
        self.__costdict = \
            {k:1.0*v/self.__divisorcount for k, v in self.__costdict.items()}
        self.__divided = True

    def strCost(self, prefix=''):
        printstr = ""
        if  self.__divided:
            pfx = prefix
        else:
            pfx = "batch_" + prefix

        for k, v in self.__costdict.items():
            printstr += (" %s%s: %7.3f" % (pfx, k, v))
        printstr = printstr.strip()

        return printstr

    def strNewCost(self, prefix=''):
        printstr = ""
        pfx = "batch_" + prefix

        for k, v in self.__newcosts.items():
            printstr += (" %s%s: %7.3f" % (pfx, k, v))
        printstr = printstr.strip()

        return printstr

    def getValues(self, prefix=''):
        printstr = ""
        pfx = "batch_" + prefix
        arr = []
        for k, v in self.__costdict.items():
            arr.append(v)

        return arr

    def updateCost(self):
        for k, v in self.__newcosts.items():
            self.__costdict[k] += v

    def multiplyScale(self, scale):
        self.__newcosts = {k:1.0*v*scale for k, v in self.__newcosts.items()}
