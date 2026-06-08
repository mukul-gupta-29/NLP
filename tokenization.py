from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
from nltk.tokenize import TreebankWordTokenizer
import spacy


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		tokenizedText = [sentence.split() for sentence in text]

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None
		tokenizer = TreebankWordTokenizer()
		tokenizedText = [tokenizer.tokenize(sentence) for sentence in text]

		return tokenizedText



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		nlp = spacy.load("en_core_web_sm")
		for sentence in text:
			doc = nlp.make_doc(sentence)
			tokens = [token.text for token in doc]
			tokenizedText.append(tokens)

		return tokenizedText
