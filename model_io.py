import sys, os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import text, sequence
from tabulate import tabulate
import pickle

nb = False
list_classes = np.array(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

print '======================================='
print '\nWelcome to the Toxicity Detector.\n'
print '======================================='

"""
REPL for determining which model to use and setting the corresponding network constants
"""
while 1:

	# prompt for input
	model_type = raw_input("Which model do you want to use? (simple LSTM/complex LSTM/naive Bayes)\n")
	model_type = str.lower(model_type)

	# check which kind of model was input and set corresponding network constants

	if model_type == "simple lstm\n" or model_type == "simple lstm":
		print ''
		model = load_model('simple_lstm_weights.h5')
		maxlen = 100
		max_features = 20000
		break

	elif model_type == "complex lstm\n" or model_type == "complex lstm":
		print ''
		model = load_model('complex_lstm_weights.h5')
		maxlen = 150
		max_features = 25000
		break

	elif model_type == "naive bayes\n" or model_type == "naive bayes":
		print ''
		print 'Loading pretrained model. One moment'
		f = open("trained_nb_model.p", "r")
		log_class_prob = pickle.load(f)
		log_not_class_prob = pickle.load(f)
		log_class_word_prob = pickle.load(f)
		log_not_class_word_prob = pickle.load(f)
		log_class_genword_prob = pickle.load(f)
		log_not_class_genword_prob = pickle.load(f)
		corpus_vocab = pickle.load(f)
		f.close()
		nb = True
		print ''
		break

	else:
		print "You input a non-valid model, please try again.\n"


if not nb:
	# import training data
	train = pd.read_csv("train.csv")

	# fill in null data points
	filled_train = train["comment_text"].fillna("unknown").values

	# create a tokenizer and fit it on the training data to ensure words are classified according to their training frequency
	tokenizer = text.Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(filled_train))

"""
REPL for determining the toxicity of an input comment based on the previously loaded model
"""
while 1:

	# prompt for input
	input_str = raw_input("Input a sentence, then hit enter to see its toxicity classification and score.\nEnter an empty string to exit.\n")

	# exit case
	if input_str == '':
		break

	if (nb):
		model_label = [0] * 6

		comment_array = input_str.split()
		for class_type in range(7):
			prob_in_class = log_class_prob[class_type]
			prob_not_in_class = log_not_class_prob[class_type]
			for word in comment_array:
				if word in corpus_vocab:
					#add prob in class
					if word in log_class_word_prob[class_type]:
						prob_in_class += log_class_word_prob[class_type][word]
					else:
						prob_in_class += log_class_genword_prob[class_type]
					#add prob not in class
					if word in log_not_class_word_prob[class_type]:
						prob_not_in_class += log_not_class_word_prob[class_type][word]
					else:
						prob_not_in_class += log_not_class_genword_prob[class_type]
			# end for
			if prob_in_class > prob_not_in_class:
				if class_type == 0:
					break
				model_label[class_type-1] = 1
				
			#end if
		toxic_classes = model_label
		classes_to_print = []
		labels_to_print = np.extract(model_label, list_classes)

		if np.sum(toxic_classes) > 0:
			print "\nOur model perceived your input as toxic in the following ways:\n"
			for label in labels_to_print:
				print(label)
			print''
		else:
			print "\nOur model did not find your input toxic\n"

	else: 
		# tokenize the input string and get its length in words
		tokenized_str = text.text_to_word_sequence(input_str, lower=False)
		num_words = len(tokenized_str)

		# encode the sentence as a sequence of numbers based off the tokenizer's fitting
		seq_str = tokenizer.texts_to_sequences(tokenized_str)

		# pad the sequence to be a valid input to the RNN
		padded_str = sequence.pad_sequences(seq_str, maxlen=maxlen)

		# make a prediction
		prediction = model.predict(padded_str)

		# average the toxicities of the sentence in each category
		prediction_avgs = np.divide(np.sum(prediction, axis=0), num_words)

		# multiply the prediction averages by 100 to get a probability percentage, then do some logical indexing
		prediction_avgs = np.multiply(prediction_avgs, 100)
		toxic_classes = np.where(prediction_avgs > 15)
		classes_to_print = prediction_avgs[toxic_classes]
		labels_to_print = list_classes[toxic_classes]

		# if there were toxic classes, print a message detailing what they are
		if np.sum(toxic_classes) > 0:
			print "\nOur model perceived your input as toxic in the following ways:"
			print tabulate([classes_to_print.tolist()], headers=labels_to_print.tolist()) + '\n'
		else:
			print "\nOur model did not find your input toxic\n"

print "Thanks for using the toxicity detector.\nGoodbye!"
print '======================================='