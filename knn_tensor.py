
import csv # csv module
import tensorflow as tf # Tensorflow module

def pred(src,dst,tz,d,wtr,temp):
	print(src,dst,wtr,temp)
	training_set = []
	training_set_y = []

	with open("dataSets/TRAIN_SET"+str(src)+str(dst)+".csv","r") as file:
		reader = csv.reader(file)
		for row in reader:
			training_set.append([row[1],row[3],row[4],row[5]])
			training_set_y.append(row[6])

	
	training_set = training_set[1:]
	training_set_y = training_set_y[1:]

	testing_set =[tz,d,wtr,temp]

	# testing_set.append([src,dst,wtr,temp])
	print(testing_set)

	training_values = tf.placeholder("float",[None,len(training_set[0])])
	test_values     = tf.placeholder("float",[len(training_set[0])])

	distance = tf.reduce_sum(tf.abs(tf.add(training_values,tf.negative(test_values))),reduction_indices=1)
	prediction = tf.arg_min(distance,0)

	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		# for i in range (len(testing_set)):

		index_in_trainingset = sess.run(prediction,feed_dict={training_values:training_set,test_values:testing_set})

		print (" The prediction is %s"%(training_set_y[index_in_trainingset]),index_in_trainingset)

		training_set = []
		# training_set_y = []
		# switch to choose training set

		with open("dataSets/TRAIN_SET" + str(src) + str(dst) + ".csv", "r") as file:
			reader = csv.reader(file)
			for row in reader:
				training_set.append([row[1], row[3], row[4], row[5],row[7]])
				# training_set_y.append(row[7])

		training_set = training_set[1:]
		# training_set_y = training_set_y[1:]

		aor=training_set[index_in_trainingset][4]
		return round(float(training_set_y[index_in_trainingset]),1),aor


