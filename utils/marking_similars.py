import pandas as pd
import os
import re
import random
import time
import binascii
from bisect import bisect_right
from heapq import heappop, heappush
from tqdm import tqdm,tqdm_notebook
import parmap
import numpy as np
import pickle
from os import path


class Signature:
    def __init__(self,num_hashes=10):
        self.numHashes = num_hashes
        self.maxShingleID=2**32-1

        # We need the next largest prime number above 'maxShingleID'.
        # I looked this value up here:
        # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
        self.nextPrime = 4294967311


    # Our random hash function will take the form of:
    #   h(x) = (a*x + b) % c
    # Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
    # a prime number just greater than maxShingleID.

    # Generate a list of 'k' random coefficients for the random hash functions,
    # while ensuring that the same value does not appear multiple times in the
    # list.
    def pickRandomCoeffs(self,k):
        # Create a list of 'k' random values.
        randList = []

        while k > 0:
        # Get a random shingle ID.
            randIndex = random.randint(0, self.maxShingleID)

            # Ensure that each random number is unique.
            while randIndex in randList:
                randIndex = random.randint(0, self.maxShingleID)

            # Add the random number to the list.
            randList.append(randIndex)
            k = k - 1

        return randList

    def add_signatures(self,df):
        coeffA = self.pickRandomCoeffs(self.numHashes)
        coeffB = self.pickRandomCoeffs(self.numHashes)


        totalShingles = 0
        signatures=[]
        for index, row in tqdm(df.iterrows(), total=len(df)):
            words=row['message_text'].split(" ")
            shinglesInDoc = set()
            # For each word in the document...
            count=0
            for index in range(0, len(words) - 2):
                # Construct the shingle text by combining three words together.

                shingle = words[index] + " " + words[index + 1] + " " + words[index + 2]
                # Hash the shingle to a 32-bit integer.
                try:
                    crc = binascii.crc32(bytearray(shingle,encoding='utf-8')) & 0xffffffff
                    # Add the hash value to the list of shingles for the current document.
                    # Note that set objects will only add the value to the set if the set
                    # doesn't already contain it.
                    shinglesInDoc.add(crc)
                    count+=1
                except TypeError:
                    pass
            totalShingles = totalShingles + count

            signature = []

            # For each of the random hash functions...
            for i in range(0, self.numHashes):
                # For each of the shingles actually in the document, calculate its hash code
                # using hash function 'i'.

                # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
                # the maximum possible value output by the hash.
                minHashCode = self.nextPrime + 1

                # For each shingle in the document...
                for shingleID in shinglesInDoc:
                    # Evaluate the hash function.
                    hashCode = (coeffA[i] * shingleID + coeffB[i]) % self.nextPrime

                    # Track the lowest hash code seen.
                    if hashCode < minHashCode:
                        minHashCode = hashCode

                # Add the smallest hash code value as component number 'i' of the signature.
                signature.append(minHashCode)
            signatures.append(signature)


            #shingles_set.append(shinglesInDoc)



        print(totalShingles)
        df['signatures']=signatures
        return df
