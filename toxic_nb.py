import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import math
import pickle

### class_type (array index of classification)
# 0 - clean
# 1 - toxic
# 2 - severe toxic
# 3 - obscene
# 4 - threat
# 5 - insult
# 6 - identity hate

# import datasets
data = pd.read_csv("train.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
data["clean"] = 1-data[list_classes].max(axis=1)
list_classes.insert(0,"clean")

# split 80% training 20% val
train, validate = train_test_split(data, test_size=0.2)

training_data = train["comment_text"]
validation_data = validate["comment_text"]

training_labels = train[list_classes].values
validation_labels = validate[list_classes].values


total_comments = training_data.size

# instantiate data structures
class_comments = [0] * 7
not_class_comments = [0] * 7

class_words = [0] * 7
not_class_words = [0] * 7

class_vocab = [0] * 7
not_class_vocab = [0] * 7

class_word_freq = [dict() for x in range(7)]
not_class_word_freq = [dict() for x in range(7)]

corpus_vocab = {}

index = 0

def clean_word_formatting(word):
	formatting_to_remove = ['\n', ',' '===', '==', '[[', ']]', '\'\'\'', '\'\'']
	for chars in formatting_to_remove:
		word = word.replace(chars, '')
	return word


# loop through each word in each comment add to counts for respective classification
for comment in training_data:

	comment_array = comment.split()
	#comment_array = comment_array.replace(/\n/g, '')
	
	comment_label = training_labels[index]

	class_type = 0
	for label in comment_label:
		if label == 1:
			class_comments[class_type] += 1
			for word in comment_array:
				clean_word_formatting(word)
				class_words[class_type] += 1
				if word in class_word_freq[class_type]:
					class_word_freq[class_type][word] += 1
				else:
					class_word_freq[class_type][word] = 1
					class_vocab[class_type] += 1
				if word not in corpus_vocab:
					corpus_vocab[word] = None
		else:
			not_class_comments[class_type] += 1
			for word in comment_array:
				clean_word_formatting(word)
				not_class_words[class_type] += 1
				if word in not_class_word_freq[class_type]:
					not_class_word_freq[class_type][word] += 1
				else:
					not_class_word_freq[class_type][word] = 1
					not_class_vocab[class_type] += 1
				if word not in corpus_vocab:
					corpus_vocab[word] = None
		#end if			
		class_type += 1


	#end for
	index += 1
#end for

# smoothing term
alpha = .0001

# prob of comment classification over corpus in general
log_class_prob = [0] * 7
log_not_class_prob = [0] * 7

# prob of word given class
log_class_word_prob = [dict() for x in range(7)]
log_not_class_word_prob = [dict() for x in range(7)]

# prob for words not found in training dataset given smoothing
log_class_genword_prob = [0] * 7
log_not_class_genword_prob = [0] * 7

for class_type in range(7):
	# compute class prob
	prob_class = float(class_comments[class_type])/float(total_comments)
	prob_not_class = 1 - prob_class
	log_class_prob[class_type] = math.log(prob_class)
	log_not_class_prob[class_type] = math.log(prob_not_class)

	# compute word given class prob
	class_word_norm = class_words[class_type] + (class_vocab[class_type] * alpha)

	for word in class_word_freq[class_type]:
		freq = class_word_freq[class_type][word]
		log_word_prob = (math.log(float(freq + alpha)/float(class_word_norm)))
		log_class_word_prob[class_type][word] = log_word_prob

	not_class_word_norm = not_class_words[class_type] + (not_class_vocab[class_type] * alpha)

	for word in not_class_word_freq[class_type]:
		freq = not_class_word_freq[class_type][word]
		log_word_prob = (math.log(float(freq + alpha)/float(not_class_word_norm)))
		log_not_class_word_prob[class_type][word] = log_word_prob

	# compute prob for words not in training data
	log_class_genword_prob[class_type] = (math.log(float(alpha)/float(class_word_norm)))
	log_not_class_genword_prob[class_type] = (math.log(float(alpha)/float(not_class_word_norm)))

# write a file
f = open("trained_nb_model.p", "w")
pickle.dump(log_class_prob, f)
pickle.dump(log_not_class_prob, f)
pickle.dump(log_class_word_prob, f)
pickle.dump(log_not_class_word_prob, f)
pickle.dump(log_class_genword_prob, f)
pickle.dump(log_not_class_genword_prob, f)
pickle.dump(corpus_vocab, f)
f.close()

# end training

####################

# begin validation
correct = 0

total_validation = validation_data.size

index = 0

# loop though comments finding prob word given classification
for comment in validation_data:

	model_label = [0] * 7

	comment_array = comment.split()

	class_type = 0

	prob_in_class = log_class_prob[class_type]
	prob_not_in_class = log_not_class_prob[class_type]

	for class_type in range(7):
		prob_in_class = log_class_prob[class_type]
		prob_not_in_class = log_not_class_prob[class_type]
		for word in comment_array:
			if word in corpus_vocab:
				clean_word_formatting(word)
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
			model_label[class_type] = 1
			if class_type == 0:
				break
				# clean is exclusive to other class
		#end if
	# end for

	l = 0
	for label in model_label:
		if label == validation_labels[index][l]:
			correct += 1
		l += 1	

	index += 1

#end for

# print accuracy as decimal 
print("Overall:" + str(float(correct)/float(total_validation*7)))



#EOF
