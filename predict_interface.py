import tensorflow as tf
import input_data
import cv2
import numpy as np
from scipy import ndimage
import sys
import os

class PredSet(object):
	def __init__(self, location, top_left=None, bottom_right=None, actual_w_h=None, prob_with_pred=None):
		self.location = location
			
		if top_left is None:
			top_left = []
		else:
			self.top_left = top_left
			
		if bottom_right is None:
			bottom_right = []
		else:
			self.bottom_right = bottom_right
		
		if actual_w_h is None:
			actual_w_h = []
		else:
			self.actual_w_h = actual_w_h
		
		if prob_with_pred is None:
			prob_with_pred = []
		else:
			self.prob_with_pred = prob_with_pred
	
	def get_location(self):
		return self.location
	
	def get_top_left(self):
		return self.top_left
	
	def get_bottom_right(self):
		return self.bottom_right
	
	def get_actual_w_h(self):
		return self.actual_w_h
	
	def get_prediction(self):
		return self.prob_with_pred[1]
	
	def get_probability(self):
		return self.prob_with_pred[0]


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    print(cy,cx)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


def pred_from_img(image, train):
	image = image
	train = train
	"""
	a placeholder for our image data:
	None stands for an unspecified number of images
	784 = 28*28 pixel
	"""
	x = tf.placeholder("float", [None, 784])

	# we need our weights for our neural net
	W = tf.Variable(tf.zeros([784,10]))
	# and the biases
	b = tf.Variable(tf.zeros([10]))

	"""
	softmax provides a probability based output
	we need to multiply the image values x and the weights
	and add the biases
	(the normal procedure, explained in previous articles)
	"""
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	"""
	y_ will be filled with the real values
	which we want to train (digits 0-9)
	for an undefined number of images
	"""
	y_ = tf.placeholder("float", [None,10])

	"""
	we use the cross_entropy function
	which we want to minimize to improve our model
	"""
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))

	"""
	use a learning rate of 0.01
	to minimize the cross_entropy error
	"""
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


	# image = sys.argv[1]
	# train = False if len(sys.argv) == 2 else sys.argv[2]
	checkpoint_dir = "cps/"

	saver = tf.train.Saver()
	sess = tf.Session()
	# initialize all variables and run init
	sess.run(tf.initialize_all_variables())
	if train:
		print("TRAIN!!!")
		# create a MNIST_data folder with the MNIST dataset if necessary
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

		# use 1000 batches with a size of 100 each to train our net
		for i in range(1000):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			# run the train_step function with the given image values (x) and the real output (y_)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		saver.save(sess, checkpoint_dir+'model.ckpt')
		"""
		Let's get the accuracy of our model:
		our model is correct if the index with the highest y value
		is the same as in the real digit vector
		The mean of the correct_prediction gives us the accuracy.
		We need to run the accuracy function
		with our test set (mnist.test)
		We use the keys "images" and "labels" for x and y_
		"""
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	else:
		# Here's where you're restoring the variables w and b.
		# Note that the graph is exactly as it was when the variables were
		# saved in a prior training run.
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print('No checkpoint found')
			exit(1)

		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



	if not os.path.exists("img/" + image + ".png"):
		print("File img/" + image + ".png doesn't exist")
		exit(1)

	# read original image
	color_complete = cv2.imread("img/" + image + ".png")

	print(("read", "img/" + image + ".png"))
	# read the bw image
	gray_complete = cv2.imread("img/" + image + ".png", 0)

	# better black and white version
	_, gray_complete = cv2.threshold(255-gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	if not os.path.exists("pro-img"):
		os.makedirs("pro-img")
	
	cv2.imwrite("pro-img/compl.png", gray_complete)

	digit_image = -np.ones(gray_complete.shape)

	height, width = gray_complete.shape

	predSet_ret = []

	"""
	crop into several images
	"""
	for cropped_width in range(100, 300, 20):
		for cropped_height in range(100, 300, 20):
			for shift_x in range(0, width-cropped_width, int(cropped_width/4)):
				for shift_y in range(0, height-cropped_height, int(cropped_height/4)):
					gray = gray_complete[shift_y:shift_y+cropped_height,shift_x:shift_x + cropped_width]
					if np.count_nonzero(gray) <= 20:
						 continue

					if (np.sum(gray[0]) != 0) or (np.sum(gray[:,0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:,
																												-1]) != 0):
						continue

					top_left = np.array([shift_y, shift_x])
					bottom_right = np.array([shift_y+cropped_height, shift_x + cropped_width])

					while np.sum(gray[0]) == 0:
						top_left[0] += 1
						gray = gray[1:]

					while np.sum(gray[:,0]) == 0:
						top_left[1] += 1
						gray = np.delete(gray,0,1)

					while np.sum(gray[-1]) == 0:
						bottom_right[0] -= 1
						gray = gray[:-1]

					while np.sum(gray[:,-1]) == 0:
						bottom_right[1] -= 1
						gray = np.delete(gray,-1,1)

					actual_w_h = bottom_right-top_left
					if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]+1) >
								0.2*actual_w_h[0]*actual_w_h[1]):
						continue

					print("------------------")
					print("------------------")

					rows,cols = gray.shape
					compl_dif = abs(rows-cols)
					half_Sm = int(compl_dif/2)
					half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
					if rows > cols:
						gray = np.lib.pad(gray,((0,0),(half_Sm,half_Big)),'constant')
					else:
						gray = np.lib.pad(gray,((half_Sm,half_Big),(0,0)),'constant')

					gray = cv2.resize(gray, (20, 20))
					gray = np.lib.pad(gray,((4,4),(4,4)),'constant')


					shiftx,shifty = getBestShift(gray)
					shifted = shift(gray,shiftx,shifty)
					gray = shifted

					cv2.imwrite("pro-img/"+image+"_"+str(shift_x)+"_"+str(shift_y)+".png", gray)

					"""
					all images in the training set have an range from 0-1
					and not from 0-255 so we divide our flatten images
					(a one dimensional vector with our 784 pixels)
					to use the same 0-1 based range
					"""
					flatten = gray.flatten() / 255.0


					print("Prediction for ",(shift_x, shift_y, cropped_width))
					print("Pos")
					print(top_left)
					print(bottom_right)
					print(actual_w_h)
					print(" ")
					prediction = [tf.reduce_max(y),tf.argmax(y,1)[0]]
					pred = sess.run(prediction, feed_dict={x: [flatten]})
					print(pred)
					
					predSet_ret.append(PredSet((shift_x, shift_y, cropped_width),
											    top_left,
												bottom_right,
												actual_w_h,
												pred))


					digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]] = pred[1]

					cv2.rectangle(color_complete,tuple(top_left[::-1]),tuple(bottom_right[::-1]),color=(0,255,0),thickness=5)

					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(color_complete,str(pred[1]),(top_left[1],bottom_right[0]+50),
								font,fontScale=1.4,color=(0,255,0),thickness=4)
					cv2.putText(color_complete,format(pred[0]*100,".1f")+"%",(top_left[1]+30,bottom_right[0]+60),
								font,fontScale=0.8,color=(0,255,0),thickness=2)


	cv2.imwrite("pro-img/"+image+"_digitized_image.png", color_complete)
	return predSet_ret



