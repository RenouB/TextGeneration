import lm
import corpus as cp

def start():
	n = None
	print('''Welcome. Let\'s create a language model together.\nWhat size n-grams do you desire?''')
	while n == None:		
		try:
			n = int(input())
			if n < 1 or n > 6:
				print('Please enter an integer between 1 and 6.')
				n = None
		except:
			print('Please enter an integer between 1 and 6.')
	mdl = lm.LanguageModel(n)
	print('''\nGreat choice!! You must train the model.\nPlease tell us the path to a text that you\'d like to use for training.''')
	filename = input()
	tokens = cp.open_file(filename)
	while tokens == None:
		print('Try again.')
		filename = input()
		tokens = cp.open_file(filename)
	mdl.train(tokens)
	print('\nYour model has been created. Here\'s a list of commands you can use to explore further.\n')
	help()
	return mdl


def help():
	print('\nkn : evaluate the probability of a text using Kneser - Ney smoothing.')
	print('lp : evaluate the probability of a text using LaPlace smoothing.')
	print('generate5 : generates 5 texts and saves them as a text file.')
	print('generate : the statistical properties of the corpus are used to generate a new text')
	print('train : add a new text to the training data')
	print('plot : plot the evolution of perplexity of the model\'s evaluation of a text')
	print('help')
	print('quit\n')


def kn(mdl):
	print('\nPlease input path to the text you want to evaluate.')
	inpt = input()
	text = cp.open_file(inpt)
	if text != None and len(text) + 1 < mdl.n:
			print('\nThis text is too small. It must have at least %s tokens.' % (mdl.n - 1))
			text = None
	while text == None:
		print('\nTry another text file? Or quit?')
		inpt = input()
		if inpt == 'quit':
			return
		text = cp.open_file(inpt)

	prob = mdl.kn_evaluate(text)
	perp = mdl.perplexity(len(text) + 1, prob)
	print('The model predicts a log probability of:\n', str(prob))
	print('The perplexity of this prediction is:\n', str(perp))


def lp(mdl):
	print('\nPlease input path to the text you want to evaluate.')
	inpt = input()
	text = cp.open_file(inpt)
	if text != None and len(text) + 1 < mdl.n:
			print('\nThis text is too small. It must have at least %s tokens.' % (mdl.n - 1))
			text = None

	while text == None:
		print('\nTry another text file? Or quit?')
		inpt = input()
		if inpt == 'quit':
			return
		text = cp.open_file(inpt)

	prob = mdl.laplace_evaluate(text)
	perp = mdl.perplexity(len(text) + 1, prob)
	print('\nThe model predicts a log probability of:\n', str(prob))
	print('The perplexity of this prediction is:\n', str(perp))


def generate(mdl):
	print('\n' + cp.detokenize(mdl.generate()))


def generate5(mdl):
	with open('new_shakespeare.txt', 'w') as f:
		for i in range(5):
			f.write(cp.detokenize(mdl.generate()))
			f.write('\n\n')


def train(mdl):
	print('\nPlease input path to the text you want to train from.')
	inpt = input()
	text = cp.open_file(inpt)
	while text == None:
		print('\nTry another text file? Or quit?')
		inpt = input()
		if inpt == 'quit':
			return
		text = cp.open_file(inpt)

	mdl.train(text)
	print('\nThe model has been updated.')
	return mdl


def plot(mdl):
	print('\nPlease input path to the text you want to plot.')
	inpt = input()
	text = cp.open_file(inpt)
	if text != None and len(text) + 1 < mdl.n:
			print('\nThis text is too small. It must have at least %s tokens.' % (mdl.n - 1))
			text = None

	while text == None:
		print('\nTry another text file? Or quit?')
		inpt = input()
		if inpt == 'quit':
			return
		text = cp.open_file(inpt)

	mdl.plot_perplexity(inpt, text)


def during(mdl):
	while True:
		ipt = input()
		while ipt not in ['kn', 'lp', 'generate', 'train', 'plot', 'quit', 'help', 'generate5']:
			print('Unknown command, try again')
			ipt = input()
		if ipt == 'kn':
			kn(mdl)
		elif ipt == 'lp':
			lp(mdl)
		elif ipt == 'generate':
			generate(mdl)
		elif ipt == 'generate5':
			generate5(mdl)
		elif ipt == 'train':
			mdl = train(mdl)
		elif ipt == 'help':
			help()
		elif ipt == 'plot':
			plot(mdl)
		elif ipt== 'quit':
			return
		print('\nWhat next?')


def main():
	mdl = start()
	during(mdl)
	return

main()