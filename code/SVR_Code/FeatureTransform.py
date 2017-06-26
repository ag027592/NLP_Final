import csv
import os
import sklearn
import gensim
import collections
import pandas
import numpy
from operator import add
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

## 'jieba' or 'JJ'
TYPE_NAME = 'jieba'
## 'fast_vec', 're_fast_vec', or 'other'
MODEL_TYPE = 'other'


if MODEL_TYPE == fast_vec":
	model = KeyedVectors.load_word2vec_format('wiki.zh_classical/wiki.zh_classical.vec', encoding='UTF-8')
elif MODEL_TYPE == re_fast_vec":
	model = KeyedVectors.load_word2vec_format('zh_f/zh.vec', encoding='UTF-8')
elif MODEL_TYPE == other":
	model = Word2Vec.load('zh/zh.bin')



## Function to implement feature transformation
def featureTransform(data_set):

	pos_dict = []
	pos_file = open('NTUSD_positive_unicode.txt', 'r', encoding = 'UTF-8')
	for row in pos_file:
		row = row.replace('\n', '')
		pos_dict += row.split(' ')
		
	neg_dict = []
	neg_file = open('NTUSD_negative_unicode.txt', 'r', encoding = 'UTF-8')
	for row in neg_file:
		row = row.replace('\n', '')
		neg_dict += row.split(' ')
	
	result = []
	for file_data in data_set:
		tmp_result = []
		for line in file_data:
			feature = []
			pos_cnt, neg_cnt = 0, 0
			for w in line:
				if not w in model:
					feature += list(numpy.zeros((1,302)))
					continue
					'''
					if len(w) < 2:
						feature += list(numpy.zeros((1,302)))
						continue
					for idx in range(len(w)):
						if not w[idx] in model: 
							feature += list(numpy.zeros((1,302)))
							continue
						tmp_w_feature = list(model[w[idx]])
						tmp_emotion_feature = [0,0]
						if w[idx] in pos_dict:
							tmp_emotion_feature = [1]
						if w[idx] in neg_dict:
							tmp_emotion_feature = [1]
						feature += [tmp_w_feature + tmp_emotion_feature]
					'''
				else:
					w_feature = list(model[w])
					tmp_emotion_feature = [0,0]
					if w in pos_dict:
						tmp_emotion_feature[0] = [1]
					if w in neg_dict:
						tmp_emotion_feature[1] = [1]
					feature += [w_feature + tmp_emotion_feature]
				#if w in pos_dict:	pos_cnt += 1
				#if w in neg_dict:	neg_cnt += 1
			#tmp_result += [feature + [pos_cnt, neg_cnt]]
			tmp_result += [feature]
		result += [tmp_result]
	
	return result

## Main function	   
idx = -2
actor_show_list = open('NTUA_Self_Label_Rated_from.csv', 'r')
dataset = []
legal_path = []
legal_file_name = []
tidied_file_name = []
for row in csv.reader(actor_show_list):
	## determine whether this actor plays this show
	idx += 1
	if idx == -1:
		continue
	
	## setting the file paths
	tmp_name = row[0]
	tmp = tmp_name.split(' ')
	path = tmp[0] + '/' + tmp[1] + '/' 
	file_name = '_'.join(tmp)
	f_path = 'features/notTransform_' + TYPE_NAME + '/' + path + file_name + '.txt'
	## determine whether this actor plays this show
	if not os.path.exists(f_path):
		#print(f_path)
		continue
	legal_path += [path]
	legal_file_name += [file_name]
	tidied_file_name += [tmp[1] + '_' + tmp[2] + '_' + tmp[0]]
	
	## fetch data from each data files
	data_file = open(f_path, 'r', encoding = 'UTF-8')
	tmp_data = []
	for line in data_file.readlines():
		line = line.replace(u'\ufeff', '')
		line = line.replace('\n', '')
		tmp_data += [line.split(',')]
	dataset += [tmp_data]


## feature transformation
features = featureTransform(dataset)
#for f in features:
#	print(f[0][0])
#	break
## store the feature


for i in range(len(legal_path)):
	feature_file = open('features/Transform_' + TYPE_NAME + '/' + legal_path[i] + legal_file_name[i] + '.txt', 'w', encoding = 'UTF-8')
	for j in range(len(features[i])):
		## j-th sentence in i-th file
		tmp_feature = [str(f) for f in features[i][j]]
		this_feature = ','.join(tmp_feature)
		feature_file.write(this_feature + '\n')
	feature_file.close()

	
## store the data with only labels
for i in range(len(legal_path)):
	for j in range(len(features[i])):
		if not os.path.exists('tidied_features/group_' + TYPE_NAME + '/' + legal_path[i]):
			os.makedirs('tidied_features/group_' + TYPE_NAME + '/' + legal_path[i])
		#tidied_feature_file = open('tidied_features/group_' + TYPE_NAME + '/' + legal_path[i] + tidied_file_name[i] + '_' + str(j) + '.txt', 'w', encoding = 'UTF-8')
		## j-th sentence in i-th file
		#tmp_feature = [str(f) for f in features[i][j]]
		#this_feature = ','.join(tmp_feature)
		#tidied_feature_file.write(this_feature)
		#tidied_feature_file.close()
		#print(features[i][j])
		group_out = pandas.DataFrame(numpy.array(features[i][j]))
		group_out.to_csv('tidied_features/group_' + TYPE_NAME + '/' + legal_path[i] + tidied_file_name[i] + '_' + str(j) + '.csv', index = None)
		if not os.path.exists('tidied_features/total_' + TYPE_NAME):
			os.makedirs('tidied_features/total_' + TYPE_NAME)
		#tidied_feature_file = open('tidied_features/total_' + TYPE_NAME + '/' + tidied_file_name[i] + '_' + str(j).zfill(2) + '.txt', 'w', encoding = 'UTF-8')
		#tidied_feature_file.write(this_feature)
		#tidied_feature_file.close()
		total_out = pandas.DataFrame(numpy.array(features[i][j]))
		total_out.to_csv('tidied_features/total_' + TYPE_NAME + '/' + tidied_file_name[i] + '_' + str(j).zfill(2) + '.csv', index = None)

	
