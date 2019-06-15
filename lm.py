from corpus import tokenize, detokenize, open_file, punc_merge
from numpy.random import choice
import math
import matplotlib.pyplot as plt


class Model:
	''' This class is used within the Language Model class. This is helpful
	because in actuality an n-order language model also contains n-1, n-1... 0
	models, which this class represents
	'''

	def __init__(self, n):
		self.n = n
		# dictionary of histories -> words found in corpus, with counts
		self.hist_words_dct = {}
		# dct of words -> histories found in corpus, with counts, used for
		# KN smoothing
		self.word_hists_dct = {}
		# list of beginning_grams which are used as seeds for generation
		self.beginning_grams = []
		# lambdas will house interpolation coefficients for KN smoothing.
		self.lambdas = {}
		# absolute discount used in KN
		self.discount = 0.75
		# it will also be helpful to know vocab size
		self.ngram_vocab_size = 0
		# as well as the total number of ngram tokens
		self.num_ngram_tokens = 0

	def compute_ngram_vocab_size(self):
		''' computes ngram vocabulary size, directly modifying self.ngram_vocab_size
		params: none
		returns: vocabn_size, int
		'''

		# iterate over all histories and their words in order
		# to get the vocab size
		vocab_size = 0
		for hist, words in self.hist_words_dct.items():
			vocab_size += len(words)

		self.ngram_vocab_size = vocab_size

	def compute_lambdas(self):
		''' Lambda coefficients are needed for the interpolation of lower order
		models in KN smoothing. This function directly modifies self.lambdas
		params: none
		returns: nothing
		'''

		# loop through all histories in hist_words_dct and count their occurences
		for hist, words in self.hist_words_dct.items():
			# if the hist is PAD(s) we have to double the count in the dictionary to
			# get an accurate count.
			if hist == ' '.join(['PAD'] * (self.n - 1)):
				self.lambdas[hist] = (self.discount / (sum(words.values()) * 2)) * \
					len(words)
			else:
				self.lambdas[hist] = (self.discount / (sum(words.values()))) * len(words)


class LanguageModel:
	def __init__(self, n=2):
		''' initialize a Language Model of order n
		params: n, int (default == 2)
		'''

		# raise error if user inputs n = 0
		if n > 0:
			self.n = n
		else:
			raise ValueError('n must be greater than 0')

		# initialize dct of unigrams -> counts to keep track of number of tokens
		# also a dct to house probability distribution of unigrams
		self.unigrams = {}
		self.num_tokens = 0
		self.unigram_pdist = {}
		# initialize list that will contain instances of Model class, for all n
		# down to n == 2
		self.models = [Model(i) for i in range(2, self.n + 1)]
		# create discount variable that is needed for KN smoothing
		self.discount = 0.75

	def train(self, token_sequences):
		'''trains the language model
		params: token_sequences, a list of lists of tokens
		'''
		# turn token_sequences into a list of lists if that isn't already the case
		if type(token_sequences[0]) != list:
			token_sequences = [token_sequences]

		# this loop will create language models of orders 1 to n
		for i in range(self.n):
			# for each of these sub models, we beginning looping through all
			# texts in token sequences to create ngrams
			for i2 in range(len(token_sequences)):
				text = token_sequences[i2]
				# pad texts
				text = ['PAD'] * i + text + ['PAD'] * i
				# unigram model is a special situation that we handle here.
				# we want a dictionary of unigram counts, a variable representing
				# num of tokens, and also a dct representing unigram prob dist
				if i == 0:
					for token in text:
						if token in self.unigrams:
							self.unigrams[token] += 1
						else:
							self.unigrams[token] = 1
						self.num_tokens += 1

					for word, count in self.unigrams.items():
						self.unigram_pdist[word] = count / self.num_tokens

				else:
					# create some variables referencing appropriate Model variables
					# just to make typing a bit easier
					hist_words_dct = self.models[i - 1].hist_words_dct
					word_hists_dct = self.models[i - 1].word_hists_dct
					beginning_grams = self.models[i - 1].beginning_grams
					# get num ngram tokens
					self.models[i - 1].num_ngram_tokens = len(text) - (self.n - 1)
					# loop through tokens in text, create ngrams and populate dictionaries
					for i3 in range(i, len(text)):
							# we need a string to use as key in hist_words_dct, so
							# convert hist into a string
							hist = ' '.join(text[i3 - i:i3])
							# to make life easier, create this word variable
							word = text[i3]
							# find tokens that start sentences. used as seed for texts generation
							if hist[0] in '.!?' or hist.startswith('PAD'):
								beginning_grams.append(' '.join(text[i3 - i + 1: i3 + 1]))
							# series of checks to populate hist_words_dct
							if hist not in hist_words_dct:
								hist_words_dct[hist] = {word: 1}
							elif word not in hist_words_dct[hist]:
								hist_words_dct[hist][word] = 1
							else:
								hist_words_dct[hist][word] += 1

							# series of checks to populate word_hists_dct
							if word not in word_hists_dct:
								word_hists_dct[word] = {hist: 1}
							elif hist not in word_hists_dct[word]:
								word_hists_dct[word][hist] = 1
							else:
								word_hists_dct[word][hist] += 1

					# now that dictionaries are done, compute lambdas and
					# ngram vocab size
					self.models[i - 1].compute_lambdas()
					self.models[i - 1].compute_ngram_vocab_size()

	def laplace_evaluate(self, tokens):
		'''evaluate the probability of a sequence using laplace smoothing
		param: tokens, list of strings
		returns: sum(probs), a float of log probabilities
		'''
		# initialize empty list to house probabilities
		probs = []
		# if self.n == 1 we can just use the num tokens and unigrams variables to do
		# the work
		if self.n == 1:
			# loop through tokens, get prob of each
			for i in range(len(tokens)):
				# if this try clause doesn't work out, it's because the token is
				# unknown
				try:
					probs.append(math.log((self.unigrams[tokens[i]] + 1)
						/ (self.num_tokens + len(self.unigrams))))
				except:
					probs.append(math.log(1 / (self.num_tokens + len(self.unigrams))))
			return sum(probs)

		# if n > 1, we do some other stuff.
		# create model variable to make life easier
		model = self.models[-1]
		# loop through token sequence, calculate prob of every ngram it contains
		for i in range(self.n - 1, len(tokens)):
			# need to join dictionary into string so that we use it as key
			# of hist_words_dct
			hist = ' '.join(tokens[i - (self.n - 1):i])
			# naming this 'word' makes life easier
			word = tokens[i]
			# check to see if both the word and hist have already been observed in
			# corpus, then compute add-one adjusted Maximum Likelihood Estimate.
			if hist in model.hist_words_dct and word in model.hist_words_dct[hist]:
				numerator = (model.hist_words_dct[hist][word] + 1)
				denominator = sum(model.hist_words_dct[hist].values()) + \
					len(model.hist_words_dct[hist])
				probs.append(math.log(numerator / denominator))
			# If only the history is in thhe corpus...
			elif hist in model.hist_words_dct:
				denominator = sum(model.hist_words_dct[hist].values()) + \
					len(model.hist_words_dct[hist])

				probs.append(math.log(1 / denominator))
			# if neither hist nor word are in corpus, the probability becomes
			# 1 / num_ngram_tokens.
			else:
				probs.append(math.log(1 / model.num_ngram_tokens))

		# sum probs together and return
		return sum(probs)

	def kn_evaluate(self, tokens):
		'''evaluate prob of a sequence using Kneser Ney smoothing
		params: tokens, list of strings
		returns: sum(probs), a float, sum of log probabilities
		'''
		# initialize empty list to house probabilities
		probs = []
		# if self.n == 1 we can skip the recursion
		if self.n == 1:
			for i in range(len(tokens)):
				# we check to see if the token is in the corpus. if it is we can
				# compute MLE adjusted by the discount
				try:
					a = (self.unigrams[tokens[i]] - self.discount) / self.num_tokens
					b = self.discount / len(self.unigrams)
					probs.append(math.log(a + b))
				# otherwise, we cannot compute MLE and the probability becomes the
				# discount divided by the vocabulary size
				except:
					probs.append(math.log(self.discount / len(self.unigrams)))
			return sum(probs)
		# if n > 1 we proceed as follows
		# loop through token sequence, compute prob of every ngram it contains
		# through recursive call of another function
		for i in range(self.n - 1, len(tokens)):
			hist = tokens[i - (self.n - 1):i]
			word = tokens[i]

			probs.append(math.log(self.kn_recursive(self.n, hist, word)))
		# sum probs together and return
		return sum(probs)

	def kn_recursive(self, n, hist, word):
		''' recursive function used to compute prob of an ngram using Kn smoothing
		params: n, int, represents order of language model
				hist, list of strings representing prefixe of ngram
				word, string representing final word of ngram
		returns: at end of recursion, returns float representing prob of ngram
		'''

		# need to join hist into string to use as key in dictionary calls
		# we give it a different name than we did in the above functions
		# because we want to conserve the history in list form
		# for the next call of the function
		hist_joined = ' '.join(hist)
		# base case of recursion is n == 1
		if n == 1:
			model = self.models[n - 1]
			# print(model.hist_words_dct)

			# check to see if word exists in our corpus, compute prob accordingly
			if word in model.word_hists_dct:
				a = (len(model.word_hists_dct[word]) - self.discount) / \
					model.ngram_vocab_size
			# otherwise set a to 0
			else:
				a = 0
			return a + self.discount / len(self.unigrams)
		# the case where n == self.n is also a special situation that requires
		# particular techniques to compute prob. we must check a bunch of
		# of different conditions. the presence of hist, word in training 
		# corpus changes how the prob is calculated in different ways
		elif n == self.n: 
			model = self.models[n - 2]
			# print(model.hist_words_dct)
			if hist_joined in model.hist_words_dct and word in \
				model.hist_words_dct[hist_joined]:
				a = (model.hist_words_dct[hist_joined][word] - self.discount) / \
					sum(model.hist_words_dct[hist_joined].values())
				lamma = model.lambdas[hist_joined]
			elif hist_joined in model.hist_words_dct:
				a = 0
				lamma = model.lambdas[hist_joined]
			else:
				a = 0
				lamma = self.discount
			return a + lamma * self.kn_recursive(n - 1, hist, word)
		# the case where 1 < n < self.n also requires its own methods
		# and we again need to check for existence of word, hist in dicts
		else:
			model = self.models[n - 1]
			#print(model.hist_words_dct)
			if hist_joined in model.hist_words_dct and word in model.word_hists_dct:
				a = (len(model.word_hists_dct[word]) - self.discount) / \
					model.ngram_vocab_size
				lamma = self.models[n - 2].lambdas[' '.join(hist[1:])]
			elif hist_joined not in model.hist_words_dct and word in \
				model.word_hists_dct:
				a = (len(model.word_hists_dct[word]) - self.discount) / \
					model.ngram_vocab_size
				lamma = self.discount
			elif hist_joined in model.hist_words_dct and \
				word not in model.word_hists_dct:
				a = 0
				lamma = self.models[n - 2].lambdas[' '.join(hist[1:])]
			else:
				a = 0
				lamma = self.discount

			return a + lamma * self.kn_recursive(n - 1, hist[1:], word)

	def perplexity(self, n, log_prob):
		''' compute perplexity of a probability prediction
		params: n, an int representing length of predicted sequence
				log_prob, a float representing log prob of sequence
		returns: float representing perplexity
		'''
		return math.e**(-(1 / n) * log_prob)

	def p_next(self, tokens):
		''' find probability distribution of next word. used to generate texts
		params: tokens, list of strings
		returns: dct representing prob dist of next word
		'''

		# if n is 1, return unigram_pdist variable
		if self.n == 1:
			return self.unigram_pdist

		# making life easier
		model = self.models[-1]
		# make sure tokens is a list
		if type(tokens) != list:
			tokens = [tokens]

		# make sure every item of that list is a string
		if set([type(item) for item in tokens]) != {str}:
			print('This method requires a list of tokens.')
			return None

		# initialize dict
		p_next = {}

		# create hist variable to represent history of ngram
		hist = ' '.join(tokens[len(tokens) - (self.n - 1): len(tokens)])
		# create variable representing denominator in MLE computation
		denominator = sum(model.hist_words_dct[hist].values())
		# compute probs of all words that the history has been observed with
		# in training corpus
		for word, count in model.hist_words_dct[hist].items():
			# According to project specifications, 'PAD' has to be represented as
			# None
			if word == 'PAD':
				p_next[None] = count / denominator
			else:
				p_next[word] = count / denominator

		return p_next

	def generate(self):
		''' method automatically generates a text based on statistical properties
		of training corpus 
		returns: generated text as string
		'''

		# if n is 1 just return a choice from the unigram_pdist variable
		if self.n == 1:
			tokens = [str(choice(list(self.unigram_pdist.keys()), 1,
				list(self.unigram_pdist.values()))[0])]

		# otherwise, choose tokens from the beginning_grams variable as seed
		else:
			model = self.models[-1]
			tokens = str(choice([item for item in model.beginning_grams])).split()
		# randomly choose a number of sentences
		num_sentences = choice(range(1, 10))
		eos = '.?!'
		# initialize an eos count to keep track of sentences
		eos_count = 0
		# begin creating sentences until we've hit our targer
		while eos_count < num_sentences:
			p_next = self.p_next(tokens)
			# here I implemented a few ad hoc, last minute solutions
			# to get this function to meet specifications
			if list(p_next.keys()) == [None]:
				tokens.append('.')
				return tokens

			next_token = 'None'

			while next_token == 'None':
				next_token = str(choice(list(p_next.keys()), 1, list(p_next.values()))[0])

			tokens.append(next_token)

			if next_token in eos:
				eos_count += 1

		# return tokens
		return tokens


	def plot_perplexity(self, title, tokens):
		''' used to plot how perplexity changes with respect to the length of a
		sequence, comparing laplace and kneser ney smoothing
		params: title, string, title of plot
				tokens: list of strings
		'''

		# initialize many lists that we'll be appending to
		tokens += ['PAD']
		lp_perplexities = [0]
		kn_perplexities = [0]
		lp_probabilities = [0]
		kn_probabilities = [0]
		# begin looping through each ngram in the token list
		for i in range(len(tokens) - (self.n - 1)):
			# create ngram variable
			ngram = tokens[i: i + self.n]
			# calculate the probability of current ngram using lp and kn
			kn_eval = self.kn_evaluate(ngram)
			lp_eval = self.laplace_evaluate(ngram)
			# add the prob of current ngram to the last element of
			# probabilities list, append new result to list
			lp_probabilities.append(lp_probabilities[-1] + lp_eval)
			kn_probabilities.append(kn_probabilities[-1] + kn_eval)
			# calculate the perplexity of this newly appended element
			# and append to perplexities lists
			lp_perplexities.append(self.perplexity(i + self.n, lp_probabilities[-1]))
			kn_perplexities.append(self.perplexity(i + self.n, kn_probabilities[-1]))

		# plot using log scale
		plt.plot(lp_perplexities[1:], label='La Place')
		plt.plot(kn_perplexities[1:], label='Kneser Ney')
		plt.legend(loc='lower right')
		plt.yscale('log')
		plt.xscale('log')
		plt.title(title)
		plt.xlabel('Text Length')
		plt.ylabel('Perplexity')
		plt.show()