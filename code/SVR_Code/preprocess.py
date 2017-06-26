## Preprocess of the raw data and the labels

import csv
import jieba
import numpy
import os
import requests

## People who label the data
DET_PEOPLE = ['chang', 'peng', 'Tsai', 'wu']
TYPE_NAME = 'jieba'

## set jieba dictionary
jieba.initialize()
jieba.set_dictionary('dict.txt.big')


## Function defined to cutting sentence with jieba
def Cutting(path, type):
	result = []
	file = open(path, 'r', encoding = 'UTF-8')
	for line in file.readlines():
		line = line.replace(u'\ufeff', '')
		line = line.replace('\n', '')
		row_data = line.split('\t')
		if type == 'jieba':
			seg_list = jieba.cut(row_data[2])
		elif type == 'JJ':
			r = requests.post('http://formosa3.nchc.org.tw:50091?pos=1', \
				json={"data": [row_data[2]]})
			tmp_seg_list = r.json()['result'][0]
			seg_list = [l[0] for l in tmp_seg_list]
		tmp_list = [w for w in seg_list if w != ' ']
		result += [[row_data[0], row_data[1], tmp_list]]
	file.close()
	return result

##	calculate the average of each sentence
##	The two input arguments are both list
def CountScore(paths, time, type):
	means = []
	num = 0.0
	for i in range(4):
		## extract values
		idx = 0
		tmp_mean = []
		if not os.path.isfile(paths[i]): continue
		num += 1.0
		file = open(paths[i], 'r')
		for row in file.readlines():
			data_list = row.split(',')
			idx += 1
			if idx == 1: continue
			if data_list[8] != '#':
				if type == 'V':
					tmp_mean += [float(data_list[8])]
				elif type == 'A':
					tmp_mean += [float(data_list[9])]
			else:
				tmp_mean += [0]
		tmp_mean = movingaverage(tmp_mean, 48).tolist()
		if i == 0:
			means += tmp_mean
		else:
			for i in range(len(means)):
				means[i] += tmp_mean[i]
		file.close()
	
	## average for 4 guys
	if num == 0.0:
		return 100
	means = [m/num for m in means]
	return CountScoreTimePeriod(means, time)

	
##	marking process
def movingaverage(interval, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')
	

##	Count average score at the time period				
def CountScoreTimePeriod(scores, time):
	mean_value = []
	for t in time:
		start = round(t[0]*10)-1
		if start < 0:
			start = 0
		end = round(t[1]*10)
		means = scores[start: end]
		mean_value += [numpy.mean(means)]

	return mean_value
				

## Main function	   
idx = 0
actor_show_list = open('NTUA_Self_Label_Rated_from.csv', 'r')
for row in csv.reader(actor_show_list):
	idx += 1
	if idx == 1:
		continue
	
	## setting the file paths
	tmp_name = row[0]
	tmp = tmp_name.split(' ')
	path = tmp[0] + '/' + tmp[1] + '/' 
	file_name = '_'.join(tmp) + '.txt'
	pure_file_name = '_'.join(tmp)
	script_path = 'script Label/' + path + file_name
	## determine whether this actor plays this show
	if not os.path.exists('script Label/' + path):
		print('script Label/' + path)
		continue
	arousal_score_paths = ['Data/' + name + '/Arousal_Active_Passive/' + \
					path + tmp[2] + '/' + 'FTAnalysis_tunelines.out' \
					for name in DET_PEOPLE]
	valence_score_paths = ['Data/' + name + '/Valence_Positive_Negative/' + \
					path + tmp[2] + '/' + 'FTAnalysis_tunelines.out' \
					for name in DET_PEOPLE]
	
	## Cutting chinese sentence
	seg_result = Cutting(script_path, TYPE_NAME)
	time = []
	for l in seg_result:
		time += [(float(l[0]), float(l[1]))]
	## calculate average score
	arousal_scores = CountScore(arousal_score_paths, time, 'A')
	valence_scores = CountScore(valence_score_paths, time, 'V')
	if arousal_scores == 100 or valence_scores == 100:
		continue
	
	## process the data with labels which will be saved
	final_result = list(seg_result)
	for i in range(len(arousal_scores)):
		final_result[i][2] = ','.join(final_result[i][2])
		final_result[i] += [str(arousal_scores[i]), str(valence_scores[i])]
		
	## store the data with labels
	show_file = open('ProcessedData/' + TYPE_NAME + '/' + path + file_name, 'w', encoding = 'UTF-8')
	for i in range(len(final_result)):
		line = '\t'.join(final_result[i]) + '\n'
		show_file.write(line)
	show_file.close()
	
	## store the data with only cut sentence
	feature_file = open('features/notTransform_' + TYPE_NAME +'/' + path + file_name, 'w', encoding = 'UTF-8')
	for i in range(len(final_result)):
		line = final_result[i][2] + '\n'
		feature_file.write(line)
	feature_file.close()
	
	## store the data with only cut sentence
	feature_file = open('features/notTransform2_' + TYPE_NAME +'/' + tmp[1] + '_' + tmp[0] + '_' + tmp[2] + '.txt', 'w', encoding = 'UTF-8')
	for i in range(len(final_result)):
		line = final_result[i][3] + '\t' + final_result[i][4] + '\t' + final_result[i][2] + '\n'
		feature_file.write(line)
	feature_file.close()
	
	## store the data with only labels and in individaul and only one directory
	for i in range(len(final_result)):
		## the data going to be stored
		line = final_result[i][3] + ',' + final_result[i][4]
		## store the data into individual folder
		if not os.path.exists('tidied_labels/group_' + TYPE_NAME +'/' + path + '/' + tmp[2]):
			os.makedirs('tidied_labels/group_' + TYPE_NAME +'/' + path + '/' + tmp[2])
		label_file = open('tidied_labels/group_' + TYPE_NAME +'/' + path + '/' + tmp[2] + '/' + tmp[1] + '_' + tmp[2] + '_' + tmp[0] + '_' + str(i) + '.txt', 'w', encoding = 'UTF-8')
		label_file.write(line)
		label_file.close()
		## store the data together into one folder
		if not os.path.exists('tidied_labels/total_' + TYPE_NAME):
			os.makedirs('tidied_labels/total_' + TYPE_NAME)
		label_file = open('tidied_labels/total_' + TYPE_NAME + '/' + tmp[1] + '_' + tmp[2] + '_' + tmp[0] + '_' + str(i).zfill(2) + '.txt', 'w', encoding = 'UTF-8')
		label_file.write(line)
		label_file.close()

	
	
	

	
	
