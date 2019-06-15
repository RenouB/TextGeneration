import pytest
from hypothesis import given
from hypothesis.strategies import lists, text
from lm import Model, LanguageModel
from corpus import tokenize, detokenize, punc_merge, open_file
import math

shakespeare = open_file('train_shakespeare.txt')
empty = open_file('empty.txt')
mind = open_file('amind.txt')


def test_tokenize_returns_list_of_strings():
	tokens = tokenize('this string')
	assert type(tokens) == list
	for item in tokens:
		assert type(item) == str


@pytest.mark.parametrize('string, tokens',
	[('This is a string.', ['this', 'is', 'a', 'string', '.']),
	 ('That isn\'t what\'s on-the-go!',
	 	['that', 'is', "n't", 'what', "'s", 'on-the-go', '!']),
	 ("Where is thy leather apron and thy rule? What dost thou with thy best apparel on? You, sir, what trade are you?",
	 ['where', 'is', 'thy', 'leather', 'apron', 'and', 'thy', 'rule', '?', 'what',
	 	'dost', 'thou', 'with', 'thy', 'best', 'apparel', 'on', '?', 'you', ',',
	 	'sir', ',', 'what', 'trade', 'are', 'you', '?']),
	 ])
def test_tokenize_produces_correct_tokens(string, tokens):
	assert tokenize(string) == tokens


@given(text=text(min_size=1))
def test_tokenize_handles_arbitrary_texts(text):
	tokens = tokenize(text)
	assert (tokens == None) or tokens


# test open_file
def test_open_handles_non_utf8_files():
	assert open_file('pdf.pdf') is None
	

def test_open_handles_empty_files():
	assert open_file('empty.txt') is None


# test the data created by open_file function
@given(text())
def test_data_has_no_empty_tokens_given_hypothesis(text):
	data = tokenize(text)
	if data is not None:
		for line in data:
			for token in line:
				assert len(token) != 0


@pytest.mark.parametrize('data', [(shakespeare), (empty), (mind), ])
def test_data_has_no_empty_tokens_with_parametrize(data):
	if data is not None:
		for line in data:
			for token in line:
				assert len(token) != 0


@given(text())
def test_data_has_no_empty_lists_given_hypothesis(text):
	data = tokenize(text)
	if data:
		for line in data:
			assert len(line) != 0

@pytest.mark.parametrize('data', [(shakespeare), (empty), (mind), ])
def test_data_has_no_empty_lists_given_parametrize(data):
	if data:
		for line in data:
			assert len(line) != 0


@given(lists(text(min_size=1), min_size=1))
def test_punc_merge_handles_arbitrary_tokens(tokens):
	assert punc_merge(tokens)


@pytest.mark.parametrize('tokens, merged', [
	(['this', '.', 'that', ',', 'the', '?', 'other'],
		['this.', 'that,', 'the?', 'other']),
	(['what', '!', 'that', '\'', 'this', ':', 'how', ';'],
		['what!', 'that\'', 'this:', 'how;'])])
def test_punc_merge_produces_expected(tokens, merged):
	assert punc_merge(tokens) == merged


@given(lists(text(min_size=1), min_size=1))
def test_detokenize_handles_arbitrary_texts(tokens):
	assert [detokenize(tokens)]


@pytest.mark.parametrize('tokens, detokenized', [
	(['this', 'is', 'a', 'string', '!'],
		['This is a string!']),
	(['what', '?', 'that', 'is', 'n\'t', 'true', '!'],
		['What? That isn\'t true!'])])
def test_detokenize_produces_expected_tokens(tokens, detokenized):
	assert [detokenize(tokens)] == detokenized


def test_train_creates_expected_word_hist_dict():
	lm = LanguageModel(2)
	data = open_file('kn_test.txt')
	lm.train(data)
	model = lm.models[-1]
	assert sorted(list(model.word_hists_dct.keys())) \
		== sorted(['this', 'text', 'shall', 'train', '.', 'PAD'])
	assert list(model.word_hists_dct['this'].keys()) == ['PAD']
	assert sorted(list(model.word_hists_dct['text'].keys())) \
		== sorted(['this', 'train'])
	assert list(model.word_hists_dct['shall'].keys()) == ['.']
	assert list(model.word_hists_dct['PAD'].keys()) == ['.']
	assert list(model.word_hists_dct['train'].keys()) == ['shall']
	assert list(model.word_hists_dct['.'].keys()) == ['text']


def test_train_creates_expected_hist_words_dict():
	lm = LanguageModel(2)
	data = open_file('kn_test.txt')
	lm.train(data)
	model = lm.models[-1]
	assert sorted(list(model.hist_words_dct.keys())) \
		== sorted(['PAD', 'this', 'text', 'shall', 'train', '.'])
	assert list(model.hist_words_dct['this'].keys()) == ['text']
	assert list(model.hist_words_dct['text'].keys()) == ['.']
	assert list(model.hist_words_dct['shall'].keys()) == ['train']
	assert list(model.hist_words_dct['train'].keys()) == ['text']
	assert list(model.hist_words_dct['PAD'].keys()) == ['this']
	assert sorted(list(model.hist_words_dct['.'].keys())) \
		== sorted(['PAD', 'shall'])


def test_discount():
	lm = LanguageModel(2)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert lm.discount == 0.75


def test_subsequent_training():
	lm = LanguageModel(2)
	data = open_file('kn_test.txt')
	lm.train(data)
	model = lm.models[-1]
	wh1_len = len(model.word_hists_dct)
	hw1_len = len(model.hist_words_dct)
	data = tokenize('This sample.')
	lm.train(data)
	model = lm.models[-1]
	wh2_len = len(model.word_hists_dct)
	hw2_len = len(model.hist_words_dct)
	assert wh2_len - wh1_len == 1
	assert hw2_len - hw1_len == 1
	assert sorted(list(model.word_hists_dct['.'].keys())) \
		== sorted(['text', 'sample'])
	assert sorted(list(model.hist_words_dct['this'].keys())) \
		== sorted(['text', 'sample'])


def test_models_have_correct_n():
	lm = LanguageModel(4)
	data = open_file('kn_test.txt')
	lm.train(data)
	for i in range(0, lm.n - 2):
		model = lm.models[i]
		assert model.n == i + 2


def test_models_have_correct_vocab_size():
	lm = LanguageModel(3)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert(lm.models[0].ngram_vocab_size == 7)
	assert(lm.models[1].ngram_vocab_size == 9)


def test_models_have_correct_lambda_size():
	lm = LanguageModel(4)
	data = open_file('kn_test.txt')
	lm.train(data)
	for i in range(0, lm.n - 2):
		model = lm.models[i]
		assert len(model.lambdas) == len(model.hist_words_dct)


def test_models_have_correct_beginning_grams():
	lm = LanguageModel(3)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert sorted(lm.models[0].beginning_grams) \
		== sorted(['this', 'shall', 'PAD'])
	assert sorted(lm.models[1].beginning_grams) \
		== sorted(['PAD this', 'this text', 'PAD PAD', 'shall train'])


def test_lm_has_correct_number_tokens_and_unigram_types():
	lm = LanguageModel(3)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert lm.num_tokens == 7
	assert len(lm.unigrams) == 5


def test_laplace_produces_expected_values():
	lm = LanguageModel(3)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert lm.laplace_evaluate(['this', 'shall', 'train', 'PAD']) \
		== -2.890371757896165
	assert lm.laplace_evaluate(['dog', 'text', '.', 'PAD']) \
		== (math.log(1 / 9) + math.log(1 / 2))


def test_laplace_produces_expected_values2():
	lm = LanguageModel(1)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert lm.laplace_evaluate(['text']) == math.log(3 / 12)
	assert lm.laplace_evaluate(['dog']) == math.log(1 / 12)


def test_kn_produces_expected_values():
	lm = LanguageModel(3)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert lm.kn_evaluate(['text', 'shall', 'train']) == -2.0770634192748685
	assert lm.kn_evaluate(['this', 'text', 'dog']) == -3.1656313103493887
	assert lm.kn_evaluate(['the', 'brown', 'cat']) == -2.4724841297894433


def test_kn_produces_expected_values_n4():
	lm = LanguageModel(4)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert lm.kn_evaluate(['shall', 'train', 'text', '.']) == -0.7742507185722116


def test_perplexity_produces_expected_values():
	lm = LanguageModel(3)
	data = open_file('kn_test.txt')
	lm.train(data)
	perp = round(lm.perplexity(2, math.log(0.5)), 5)
	correct = round(math.sqrt(2), 5)
	assert perp == correct


def test_p_next_sums_to_one():
	lm = LanguageModel(3)
	data = open_file('kn_test.txt')
	lm.train(data)
	assert sum(lm.p_next(['this', 'text']).values()) == 1

