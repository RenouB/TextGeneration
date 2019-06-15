from nltk.tokenize import word_tokenize

def tokenize(text):
	''' tokenizes a text using nltk.word_tokenize
	params: text, a string
	returns: tokens, a list of strings
	'''

	# check to see if text is only whitespace
	if len(text.strip()) == 0:
		return None

	tokens = word_tokenize(text.lower())

	return tokens


def open_file(file):
	''' opens and tokenizes a file
	params: text file, must be utf-8 encoded
	returns: data, a list of lists of tokens, which are strings
	'''
	try:		
		with open(file) as f:
			# ensure file is in correct encoding
			try:
				read = f.read()
			except UnicodeDecodeError as error:
				print('''File not cannot be decoded. Make sure it\'s a UTF-8 decodable text file.''')
				return None

			# ensure file is not empty
			if len(read) == 0:
				print('The text appears to be empty. Please give us some data.')
				return None

			data = tokenize(read)
	except:
		print('\nSomething went wrong.')
		return None

	return data


def punc_merge(tokens):
	''' merges a punctuation token with the token that precedes it in list
	params: tokens, a list of tokens
	returns: merged, a list of tokens with punctuation tokens
	merged into tokens that preceded them
	'''

	last = []
	merged = []
	for token in tokens:
		if (token[0] in '%!\',.:;?\\`' or token.startswith('n\'')) and len(last):
			merged += (last[:-1] + [last[-1] + token])
			last = []
		else:
			last.append(token)

	merged += last
	return merged



def detokenize(tokens):
	''' detokenizes a list of tokens, producing a text
	params: tokens, a list of strings
	returns: 
	'''
	# eliminate pads and undesireable punctuation marks
	tokens = [token for token in tokens if token != 'PAD' and token not in ')`\'t"--']
	# capitalize Is
	for i in range(len(tokens)):
		if tokens[i] == 'i':
			tokens[i] = 'I'
	# merge what's left
	merged = punc_merge(tokens)
	if len(merged) == 0:
		return None

	merged[0] = merged[0].capitalize()

	for i in range(1, len(merged)):
		if merged[i - 1][-1] in '!.?' or merged[i] == 'i':
			merged[i] = merged[i].capitalize()

	as_one_text = [' '.join(merged)]

	return ' '.join(as_one_text)