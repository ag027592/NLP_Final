import csv
import os
import pickle
from sklearn.svm import LinearSVR
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectPercentile,f_regression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.externals import joblib


TYPE_NAME = 'jieba'


## main function
idx = -2
actor_show_list = open('NTUA_Self_Label_Rated_from.csv', 'r')
dataset = []
labels_v = []
labels_a = []
emotion_team = []
for row in csv.reader(actor_show_list):
	## determine whether this actor plays this show
	idx += 1
	if idx == -1:
		continue
	## setting the file paths
	tmp_name = row[0]
	tmp = tmp_name.split(' ')
	path = tmp[0] + '/' + tmp[1] + '/' + '_'.join(tmp) + '.txt'
	f_path = 'features/Transform3_' + TYPE_NAME + '/' + path
	## determine whether this actor plays this show
	if not os.path.exists(f_path):
		#print(f_path)
		continue	
	## determine if this show has labels
	if not os.path.isfile('ProcessedData/' + TYPE_NAME + '/' + path): continue
	
	## fetch data from each data files
	data_file = open(f_path, 'r', encoding = 'UTF-8')
	tmp_data = []
	cnt = 0
	for line in data_file.readlines():
		line = line.replace(u'\ufeff', '')
		line = line.replace('\n', '')
		tmp_data += [[float(content) for content in line.split(',')]]
		cnt += 1
		emotion_team += [[tmp[0], int(tmp[1]), tmp[2]]]
	dataset += tmp_data
	data_file.close()

	label_file = open('ProcessedData/' + TYPE_NAME + '/' + path, 'r', encoding = 'UTF-8')
	tmp_arousal = []
	tmp_valence = []
	for line in label_file.readlines():
		line = line.replace(u'\ufeff', '')
		line = line.replace('\n', '')
		tmp_label = line.split('\t')
		tmp_arousal += [float(tmp_label[3])]
		tmp_valence += [float(tmp_label[4])]
	labels_a += tmp_valence
	labels_v += tmp_arousal
	label_file.close()

	

model_v = LinearSVR(C=1.0, epsilon=0.1)
model_a = LinearSVR(C=1.0, epsilon=0.1)
	
test_v = []
test_a = []
gt_label_v = []
gt_label_a = []
affection = []
## training with leave one group out
for i in range(1, 23):
	## selecting data group
	training_data = [dataset[x] for x in range(len(dataset)) if not emotion_team[x][1] == i]
	label_train_v = [labels_v[x] for x in range(len(labels_v)) if not emotion_team[x][1] == i]
	label_train_a = [labels_a[x] for x in range(len(labels_a)) if not emotion_team[x][1] == i]
	## training
	model_v.fit(training_data, label_train_v)
	model_a.fit(training_data, label_train_a)
	
	test_data = [dataset[x] for x in range(len(dataset)) if emotion_team[x][1] == i]
	#test_data_v = selectors_v.transform(test_data)
	#test_data_a = selectors_a.transform(test_data)
	## testing valence
	label_test_v = [labels_v[x] for x in range(len(labels_v)) if emotion_team[x][1] == i]
	predict = model_v.predict(test_data)
	test_v += predict.tolist()
	gt_label_v += label_test_v
	#print('count correlation_v except : ' + str(i))
	#print(spearmanr(predict.tolist(), label_test_v, axis = None))
	## testing arousal
	label_test_a = [labels_a[x] for x in range(len(labels_a)) if emotion_team[x][1] == i]
	predict = model_a.predict(test_data)
	test_a += predict.tolist()
	gt_label_a += label_test_a
	#print('count correlation_a except : ' + str(i))
	#print(spearmanr(predict.tolist(), label_test_a, axis = None))
	## data belongings
	tmp_emotion = [emotion_team[x] for x in range(len(emotion_team)) if emotion_team[x][1] == i]
	affection += tmp_emotion
	print('===================================')
	print(i)
	for idx in range(len(tmp_emotion)):
		print(tmp_emotion[idx][0] + '_' + tmp_emotion[idx][2])
		print(test_v[idx])
		print(test_a[idx])
	print('==================================')

## save the model
joblib.dump(model_v, 'model_v_' + TYPE_NAME + '.pkl')
joblib.dump(model_a, 'model_a_' + TYPE_NAME + '.pkl')

'''
## calculate the correlation of each emotion
emotions = ['angry', 'frustration', 'happy', 'neutral', 'sad', 'surprise']
for e in emotions:
	tmp_result_v = [test_v[i] for i in range(len(test_v)) if affection[i][0] == e]
	tmp_result_a = [test_a[i] for i in range(len(test_a)) if affection[i][0] == e]
	tmp_label_v = [gt_label_v[i] for i in range(len(gt_label_v)) if affection[i][0] == e]
	tmp_label_a = [gt_label_a[i] for i in range(len(gt_label_a)) if affection[i][0] == e]
	tmp_affection = [affection[i] for i in range(len(affection)) if affection[i][0] == e]
	stack_result_v = {}
	stack_result_a = {}
	stack_label_v = {}
	stack_label_a = {}
	for i in range(1, 23):
		stack_result_v[i] = []
		stack_result_a[i] = []
		stack_label_v[i] = []
		stack_label_a[i] = []
	for idx in range(len(tmp_affection)):
		stack_result_v[tmp_affection[idx][1]] += [tmp_result_v[idx]]
		stack_result_a[tmp_affection[idx][1]] += [tmp_result_a[idx]]
		stack_label_v[tmp_affection[idx][1]] += [tmp_label_v[idx]]
		stack_label_a[tmp_affection[idx][1]] += [tmp_label_a[idx]]
	print('--------------------------------------------------------------------')
	print(e)
	for idx in stack_label_v:
		if not stack_label_v[idx]: continue
		print(idx)
		print(spearmanr(stack_result_v[idx], stack_label_v[idx]))
		print(spearmanr(stack_result_a[idx], stack_label_a[idx]))
'''
		
print('count correlation : ' + str(percentile))
print(spearmanr([test_v], [gt_label_v], axis = None))
print(spearmanr([test_a], [gt_label_a], axis = None))