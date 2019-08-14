import matplotlib.pyplot as plt
import math
import numpy as np
import cv2


def plot_image(image, img_shape, plt_show, channels=3, name=""):
	'''
	Plots one image
	:param image: the image to plot
	:param img_shape: the shape of the image
	:param plt_show: debug parameter, if false it wont plot the image
	:param number of color channels, 1 for greyscale and 3 for color
	:param name: title of image default: ""
	'''
	# fig = plt.figure()
	# fig.canvas.set_window_title(name)
	# plt.suptitle(name)
	# if len(image.shape) == 2:
		# plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='gray')
	# elif image.shape[2] == 3:
		# cpy = image.reshape((img_shape[0], img_shape[1], 3))
		# cpy = cv2.cvtColor(cpy, cv2.COLOR_BGR2RGB)
		# cv2.imwrite("tmp_img/cpy.png", cpy)
		# cpy = cv2.imread("tmp_img/cpy.png")
		# plt.imshow(cpy, interpolation='nearest')
	# else:
		# cpy = image.reshape((img_shape[0], img_shape[1], 4))
	#	cpy = cv2.cvtColor(cpy, cv2.COLOR_RGBA2RGB)
		# cv2.imwrite("tmp_img/cpy.png", cpy)
		# cpy = cv2.imread("tmp_img/cpy.png")
		# cpy = cv2.cvtColor(cpy, cv2.COLOR_BGR2RGB)
		# plt.imshow(cpy, interpolation='nearest')
	# if plt_show == True:
		# plt.show()
	return


def plot_nine_images(images, classes, cls_true, plt_show, img_shape, channels, cls_pred=None, name=""):
	'''
	Plots nine images in one view
	:param images: the images to plot
	:param class_one: name to identify class one
	:param class_zero: name to identify class zero
	:param cls_true: the true classes
	:param plt_show: debug parameter, if false it wont plot the image
	:param cls_pred: the predicted classes, default: None
	:param name: title of plot, default: ""
	'''

	try:
		assert len(images) == len(cls_true) == 9
	except:
		return


	# Create figure with 3x3 sub-plots.
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Plot image.

		img = images[i]
		cpy = np.copy(img)

		if channels == 3:
			cpy = cpy.reshape((img_shape[0], img_shape[1], 3))
			cpy = cv2.resize(cpy, (100, 100))
			cpy = cv2.cvtColor(cpy, cv2.COLOR_BGR2RGB)
			cv2.imwrite("tmp_img/cpy.png", cpy)
			cpy = cv2.imread("tmp_img/cpy.png")  #not sure why it's not working without but this fixes some color isssues
			# exit()
		else:
			cpy = cpy.reshape(img_shape)
			cpy = np.resize(cpy, (100, 100))
			cv2.imwrite("tmp_img/cpy.png", cpy)
			cpy = cv2.imread("tmp_img/cpy.png")  #not sure why it's not working without but this fixes some color isssues
		ax.imshow(cpy)

		# Show true and predicted classes.
		if cls_pred is None:

			label_name = classes[cls_true[i]]
			xlabel = "True: {0}".format(label_name)
		else:
			label_name = classes[cls_true[i]]
			label_name_pred = classes[cls_pred[i]]
			xlabel = "True: {0}, Predict: {1}".format(label_name, label_name_pred)

		# Show the classes as the label on the x-axis.
		ax.set_xlabel(xlabel)

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	# Sets the title of the figure window
	plt.suptitle(name)
	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	if plt_show==True:
		plt.show()


def plot_conv_layer(layer, image, session, plt_show, x, name=""):
	'''
	@author = Magnus Erik Hvass Pedersen
	plots a convolutional layer applied to an image
	:param layer: the TF layer
	:param image: the image to plot
	:param session: TF session
	:param plt_show: debug parameter, if false it wont plot the image
	:param x: the x tf.placeholder object
	:param name: title of plot, default: ""
	'''
	# Assume layer is a TensorFlow op that outputs a 4-dim tensor
	# which is the output of a convolutional layer,
	# e.g. layer_conv1 or layer_conv2.

	# Create a feed-dict containing just one image.
	# Note that we don't need to feed y_true because it is
	# not used in this calculation.
	feed_dict = {x: [image]}

	# Calculate and retrieve the output values of the layer
	# when inputting that image.
	values = session.run(layer, feed_dict=feed_dict)

	# Number of filters used in the conv. layer.
	num_filters = values.shape[3]

	# Number of grids to plot.
	# Rounded-up, square-root of the number of filters.
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a grid of sub-plots.
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot the output images of all the filters.
	for i, ax in enumerate(axes.flat):
		# Only plot the images for valid filters.
		if i < num_filters:
			# Get the output image of using the i'th filter.
			# See new_conv_layer() for details on the format
			# of this 4-dim tensor.
			img = values[0, :, :, i]

			# Plot image.
			ax.imshow(img, interpolation='nearest', cmap='gray')

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	# Sets the title of the figure window
	fig.canvas.set_window_title(name)
	plt.suptitle(name)
	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	if plt_show == True:
		plt.show()


def plot_conv_weights(weights, session, plt_show, convolutional_layer=0, name=""):
	"""
	@author = Magnus Erik Hvass Pedersen
	plots the kernel in the convolotional layers. Strong blue = means the filter has a negative reaction to black pixels
	and strong white means the filter has a positive reaction to white pixels
	:param weights: weights to print
	:param session: the TF session
	:param plt_show: debug parameter, if false it wont plot the image
	:param convolutional_layer: the convolutional layer to print the weights from. default: 0
	:param name: title of plot, default: ""
	"""
	# Assume weights are TensorFlow ops for 4-dim variables
	# e.g. weights_conv1 or weights_conv2.

	# Retrieve the values of the weight-variables from TensorFlow.
	# A feed-dict is not necessary because nothing is calculated.
	w = session.run(weights)

	# Get the lowest and highest values for the weights.
	# This is used to correct the colour intensity across
	# the images so they can be compared with each other.
	w_min = np.min(w)
	w_max = np.max(w)

	# Number of filters used in the conv. layer.
	num_filters = w.shape[3]

	# Number of grids to plot.
	# Rounded-up, square-root of the number of filters.
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a grid of sub-plots.
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot all the filter-weights.
	for i, ax in enumerate(axes.flat):
		# Only plot the valid filter-weights.
		if i < num_filters:
			# Get the weights for the i'th filter of the input channel.
			# See new_conv_layer() for details on the format
			# of this 4-dim tensor.
			img = w[:, :, convolutional_layer, i]

			# Plot image.
			ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	# Sets the title of the figure window
	fig.canvas.set_window_title(name)
	plt.suptitle(name)

	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	if plt_show == True:
		plt.show()


def print_test_accuracy(data, batch_size, x, y_true, session, y_pred_cls, classes, plt_show, img_shape, channels, confusion_matrix=None, num_classes=7, show_example_errors=False, show_confusion_matrix=False, name=""):
	"""
	@author = Magnus Erik Hvass Pedersen
	Prints the test accuracy, options avaliable to print confusion matrix too.

	:param data: the data object
	:param batch_size: size of the batches
	:param x: the x tf.placeholder object
	:param y_true: the true classes
	:param session: session
	:param y_pred_cls: the predicted classes
	:param class_one: name to identify class one
	:param class_zero: name to identify class zero
	:param plt_show: debug parameter, if false it wont plot the image
	:param img_shape: the image shape
	:param channels: the amount of color channels, 3 for color, 1 for greyscale
	:param confusion_matrix: the confusion matrix to plot, default: None
	:param num_classes: the number of classes Default:2
	:param show_example_errors: weather or not to show example errors, default: False
	:param show_confusion_matrix: weather or not to show confusion matrix, default: False
	:param name: title of plot, default: ""

	:return: accuracy as float and confusion matrix as array
	"""
	# Number of images in the test-set.
	num_test = len(data.test.images)

	# Allocate an array for the predicted classes which
	# will be calculated in batches and filled into this array.
	cls_pred = np.zeros(shape=num_test, dtype=np.int)

	# Now calculate the predicted classes for the batches.
	# We will just iterate through all the batches.
	# There might be a more clever and Pythonic way of doing this.

	# The starting index for the next batch is denoted i.
	i = 0

	while i < num_test:
		# The ending index for the next batch is denoted j.
		j = min(i + batch_size, num_test)

		# Get the images from the test-set between index i and j.
		images = data.test.images[i:j, :]

		# Get the associated labels.
		labels = data.test.labels[i:j, :]

		# Create a feed-dict with these images and labels.
		feed_dict = {x: images,
					 y_true: labels}

		# Calculate the predicted class using TensorFlow.





		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

		# Set the start-index for the next batch to the
		# end-index of the current batch.
		i = j

	# Convenience variable for the true class-numbers of the test-set.
	cls_true = data.test.cls

	# Create a boolean array whether each image is correctly classified.
	correct = (cls_true == cls_pred)

	# Calculate the number of correctly classified images.
	# When summing a boolean array, False means 0 and True means 1.


	correct_sum = correct.sum()

	# Classification accuracy is the number of correctly classified
	# images divided by the total number of images in the test-set.
	acc = float(correct_sum) / num_test

	# Print the accuracy.
	msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
	print(msg.format(acc, correct_sum, num_test))

	# Plot some examples of mis-classifications, if desired.
	if show_example_errors:
		plot_example_errors(cls_pred, correct, data, classes, plt_show, channels=channels, img_shape=img_shape, name=name)

	cm = None
	# Plot the confusion matrix, if desired.
	if show_confusion_matrix:
		print("Confusion Matrix:")

		cm = plot_confusion_matrix(cls_pred, data, confusion_matrix, num_classes, plt_show, name=name)

	return acc, cm


def plot_confusion_matrix(cls_pred, data, confusion_matrix, num_classes, plt_show, name=""):
	"""
	@author = Magnus Erik Hvass Pedersen
	Plots and prints the confusion matrix
	:param cls_pred: the predicted classes
	:param data: the data object
	:param confusion_matrix: the confusion matrix to print
	:param num_classes: the total number of classes
	:param plt_show: debug parameter, if false it wont plot the image
	:param name: title of plot, default: ""
	:return: confusion matrix
	"""
	# This is called from print_test_accuracy().

	# cls_pred is an array of the predicted class-number for
	# all images in the test-set.

	# Get the true classifications for the test-set.
	cls_true = data.test.cls

	# Get the confusion matrix using sklearn.
	cm = confusion_matrix(y_true=cls_true,
						  y_pred=cls_pred)

	# Print the confusion matrix as text.
	print(cm)

	if plt_show:
		# Plot the confusion matrix as an image.
		plt.matshow(cm)
	return cm


def plot_example_errors(cls_pred, correct, data, classes, plt_show, img_shape, channels, name=""):
	"""
	@author = Magnus Erik Hvass Pedersen
	some of the images that were incorrectly classified

	:param cls_pred: the predicted classes
	:param correct: the correctly classified labels
	:param data: the data object
	:param class_one: the random class
	:param class_zero: the identified class
	:param plt_show: debug parameter, if false it wont plot the image
	:param img_shape: the shape of the images
	:param channels: the amount of channels, 1 = grey and 3 = color
	:param name: title of plot, default: ""
	"""
	# This function is called from print_test_accuracy() below.

	# cls_pred is an array of the predicted class-number for
	# all images in the test-set.

	# correct is a boolean array whether the predicted class
	# is equal to the true class for each image in the test-set.

	# Negate the boolean array.
	incorrect = (correct == False)

	# Get the images from the test-set that have been
	# incorrectly classified.
	images = data.test.images[incorrect]

	# Get the predicted classes for those images.
	cls_pred = cls_pred[incorrect]

	# Get the true classes for those images.
	cls_true = data.test.cls[incorrect]

	# Tries to plot the first 9 images.
	plot_nine_images(images[0:9], classes, cls_true[0:9], plt_show, channels=channels, img_shape=img_shape, name=name)


def print_prediction(data, batch_size, x, y_true, session, y_pred_cls, classes, image_shape, num_channels, plt_show):
	"""
	@author = Magnus Erik Hvass Pedersen
	Prints the predicted image from the classify class
	:param data: the data object
	:param batch_size: size of the batches
	:param x: the x tf.placeholder object
	:param y_true: the true classes
	:param session: session
	:param y_pred_cls: the predicted classes
	:param class_one: name to identify class one
	:param class_zero: name to identify class zero
	:param image_shape: the image shape
	:param plt_show: debug parameter, if false it wont plot the image
	"""
	# Number of images in the test-set.
	num_test = len(data.test.images)

	# Allocate an array for the predicted classes which
	# will be calculated in batches and filled into this array.
	cls_pred = np.zeros(shape=num_test, dtype=np.int)



	# Now calculate the predicted classes for the batches.
	# We will just iterate through all the batches.
	# There might be a more clever and Pythonic way of doing this.

	# The starting index for the next batch is denoted i.
	i = 0

	while i < num_test:
		# The ending index for the next batch is denoted j.
		j = min(i + batch_size, num_test)

		# Get the images from the test-set between index i and j.
		images = data.test.images[i:j, :]

		# Get the associated labels.
		labels = data.test.labels[i:j, :]

		# Create a feed-dict with these images and labels.
		feed_dict = {x: images,
		             y_true: labels}

		# Calculate the predicted class using TensorFlow.
		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

		# Set the start-index for the next batch to the
		# end-index of the current batch.
		i = j

	iteration = 0
	for image in data.test.images:
		msg = "Image is of type {}".format(classes[cls_pred[iteration]])
		print(msg)
		plot_image(image, image_shape, plt_show, num_channels, name=msg)
		iteration = iteration + 1


