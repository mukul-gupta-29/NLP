from util import *

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
import json
import math
from collections import Counter


class StopwordRemoval():

    def fromList(self, text):
        """
        Stopword removal using NLTK stopword list (Top-down / curated)

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens

        Returns
        -------
        list
            A list of lists with stopwords removed
        """
        stop_words = set(stopwords.words('english'))
        stopwordRemovedText = [
            [token for token in sentence if token.lower() not in stop_words]
            for sentence in text
        ]
        return stopwordRemovedText


   