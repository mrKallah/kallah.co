import cv2
import numpy as np
import random as rand
import os
import glob
import warnings
import shutil


def save(img, dir_path, name, print_interval=100):
	'''
	Saves the intput image in to a file
	:param img: the image to save
	:param dir_path: Where to save the image
	:param name: The name to save the image with an extension, EG. cat.png
	:param print_interval: how often to print. For mass printing. -1 for no printing
	'''
	# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	if print_interval != -1:
		if i % print_interval == 0:
			print("Handled {0}/{2} images, currently in path {1}".format(i, dir_path, get_total_files()))
	cv2.imwrite(os.path.join(dir_path, name), img)
	
total_files = 0
def set_total_files(new_size):
	global total_files
	total_files = new_size

def get_total_files():
	global total_files
	return total_files


def prepare_data(data_directory, file_name_identifier, image_shape, num_channels, num_augment, file_format=".thisisthefileformat", sub_dir="augmented/"):
	'''
	This function reads the data from a directory and then augments it into a subdirectory.
	It will name all files exept the ones containing the file name identifier as rand.
	It will augmet the data to num_augment amount of images.
	:param data_directory: the directory to find the files
	:param file_name_identifier: the file name to change the non-random image names to
	:param image_shape: the shape of the out put images without the color chanel
	:param num_channels: the color chanels, 1 for grey, 3 for color
	:param num_augment: this can be 1, 10, 20, 30 and 40 and will make the original image into 1-num_augment images
	:param file_format: a file format of your choise, jpeg, png and jpg are defaults
	'''

	#read all the png, jpg and jpeg and selected files in the folder
	files = glob.glob(os.path.join(data_directory, "*.{}".format(file_format)))
	files2 = glob.glob(os.path.join(data_directory, "*.{}".format("jpg")))
	files3 = glob.glob(os.path.join(data_directory, "*.{}".format("jpeg")))
	files4 = glob.glob(os.path.join(data_directory, "*.{}".format("png")))
	files5 = []
	if file_format != "png":
		files5 = glob.glob(os.path.join(data_directory, "*.{}".format(file_format)))

	# add all the files into first array
	for f in files2:
		files.append(f)
	for f in files3:
		files.append(f)
	for f in files4:
		files.append(f)
	for f in files5:
		files.append(f)

	# delete and re-create the output path to empty it
	out_path = os.path.join(data_directory, sub_dir)
	if os.path.exists(out_path):
		shutil.rmtree(out_path)

	if not os.path.exists(out_path):
		os.makedirs(out_path)
	
	set_total_files(len(files)*num_augment)
	
	# Go thorough all the files in the files array
	iter = 0
	for f in files:
		# print(f)
		# read the image as color or grey images
		if num_channels == 1:
			image = cv2.imread(f, 0)
		else:
			image = cv2.imread(f, num_channels)

		# convert the image to a numpy array of float32s
		image = np.asarray(image, dtype="float32")

		# set the name to be the identifier as default
		name = file_name_identifier

		# change the name identifier based on if it is in the name of tte file or not
		f = files[iter]
		if f.find(file_name_identifier) == -1:
			name = "rand"

		# resize the image to the propper shape
		image = resize(image, image_shape)

		# augment the images to more images
		augment_switch(image, out_path, name, num_channels, num_augment)

		# increase the current iteration
		iter = iter + 1


def augment_switch(img, out_path, name, num_channels, num_augment, print_errors=True):
	'''
	This is a switch function that makes sure that the num_augment is of correct type
	:param img: the image to augment
	:param out_path: the path to save the augmented images
	:param name: the name without the iterations or extensions, example: "rand" would be written as rand0.png... rand34.png
	:param num_channels: the number of channels, 1 for grey and 3 for color
	:param num_augment: the number of augmentations to augment to. Legal values are 1, 10, 20, 30 and 40
	:return: only returns -1 if wrong number augments
	'''
	if num_augment == 1:
		augment_to_1(img, out_path, name, num_channels)
	elif num_augment == 10:
		augment_to_10(img, out_path, name, num_channels)
	elif num_augment == 20:
		augment_to_20(img, out_path, name, num_channels)
	elif num_augment == 30:
		augment_to_30(img, out_path, name, num_channels)
	elif num_augment == 40:
		augment_to_40(img, out_path, name, num_channels)
	else:
		if print_errors:
			warnings.warn("WARNING: You can only augment to 1, 10, 20, 30 or 40 images. Please change your augment number. Defaulting to 1 augmentation")
		augment_to_1(img, out_path, name, num_channels)
		return -1


i = -1 #first time is -1+1 = 0
def iterate(change=0, to_change=False):
	'''
	This iterates either the global i variable within the class or if to_change is True then it iterates the change
	:param change: the value to iterate if to_change is True
	:param to_change: Whether or not to iterate the global i or the passed in change variable.
	:return: returns i+1 if to_change is False and change+1 if to_change is True
	'''
	global i
	if to_change:
		change = change+1
		return change
	else:
		i = i + 1
		return i


def re_iterate():
	'''
	Restarts the iteration. Sets i back to its default value
	'''
	global i
	i = -1


def resize(img, shape):
	'''
	resizes an image to the shape given. Ignores color channels. (50, 50) works for resizing both a color image and
	greyscale image to 50 by 50 and keeps the original color channels
	:param img: the image to resize
	:param shape: the new shape as a tuple
	:return: the resized image
	'''
	cpy = np.copy(img)
	cpy = cv2.resize(cpy, shape)
	return cpy


def augment_to_1(img, output_path, name, num_chanels):
	'''
	this saves the original image in the augmented path
	:param img: the image to save
	:param output_path: path to save image in
	:param name: Name of the output image
	:param num_chanels: unused, but kept to keep the 5 augment_to_x functions the same
	'''
	global i
	save(img, output_path, (name + "{}.png".format(iterate())))


def augment_to_10(img, output_path, name, num_chanels):
	'''
	this saves the original image and 9 augmentations of it in the augmented path
	:param img: the image to augment and then save
	:param output_path: path to save images in
	:param name: Name of the output images
	:param num_chanels: unused, but kept to keep the 5 augment_to_x functions the same
	'''
	global i
	save(brighten_darken(img, 0.8), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.2), output_path, (name + "{}.png".format(iterate())))
	save(blur(img), output_path, (name + "{}.png".format(iterate())))
	save(sharpen(img), output_path, (name + "{}.png".format(iterate())))

	save(add_salt_and_pepper_noise(img, 400), output_path, (name + "{}.png".format(iterate())))

	save(mirror(img), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, -1), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, 1), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 0, 0, 1), output_path, (name + "{}.png".format(iterate())))
	save(img, output_path, (name + "{}.png".format(iterate())))


def augment_to_20(img, output_path, name, num_chanels):
	'''
	this saves the original image and 19 augmentations of it in the augmented path
	:param img: the image to augment and then save
	:param output_path: path to save images in
	:param name: Name of the output images
	:param num_chanels: to not add color grains to greyscale images.
	'''
	global i
	save(affine_transform(img, num_chanels), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.8), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.9), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.1), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.2), output_path, (name + "{}.png".format(iterate())))
	save(add_light_and_color_blur(img), output_path, (name + "{}.png".format(iterate())))
	save(blur(img), output_path, (name + "{}.png".format(iterate())))
	save(sharpen(img), output_path, (name + "{}.png".format(iterate())))

	if num_chanels == 1:
		save(add_salt_and_pepper_noise(mirror(img), 200), output_path, (name + "{}.png".format(iterate())))
	else:
		save(add_tutti_frutty_noise(img, 400), output_path, (name + "{}.png".format(iterate())))

	save(pepper(img, 400), output_path, (name + "{}.png".format(iterate())))
	save(salt(img, 400), output_path, (name + "{}.png".format(iterate())))
	save(add_salt_and_pepper_noise(img, 400), output_path, (name + "{}.png".format(iterate())))

	save(mirror(img), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, -1), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, 1), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 1, 0, 0, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 1, 0, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 0, 1, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 0, 0, 1), output_path, (name + "{}.png".format(iterate())))
	save(img, output_path, (name + "{}.png".format(iterate())))


def augment_to_30(img, output_path, name, num_chanels):
	'''
	this saves the original image and 29 augmentations of it in the augmented path
	:param img: the image to augment and then save
	:param output_path: path to save images in
	:param name: Name of the output images
	:param num_chanels: to not add color grains to greyscale images.
	'''
	global i
	save(affine_transform(img, num_chanels), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.6), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.7), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.8), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.9), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.1), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.2), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.3), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.4), output_path, (name + "{}.png".format(iterate())))
	save(add_light_and_color_blur(img), output_path, (name + "{}.png".format(iterate())))
	save(blur(img), output_path, (name + "{}.png".format(iterate())))
	save(sharpen(img), output_path, (name + "{}.png".format(iterate())))

	if num_chanels == 1:
		save(add_salt_and_pepper_noise(mirror(img), 50), output_path, (name + "{}.png".format(iterate())))
	else:
		save(add_tutti_frutty_noise(img, 400), output_path, (name + "{}.png".format(iterate())))

	save(pepper(img, 400), output_path, (name + "{}.png".format(iterate())))

	save(salt(img, 400), output_path, (name + "{}.png".format(iterate())))

	save(add_salt_and_pepper_noise(img, 400), output_path, (name + "{}.png".format(iterate())))
	save(add_salt_and_pepper_noise(img, 200), output_path, (name + "{}.png".format(iterate())))
	save(add_salt_and_pepper_noise(img, 100), output_path, (name + "{}.png".format(iterate())))

	save(mirror(img), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, -2), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, -1), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, 1), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, 2), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 1, 0, 0, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 1, 0, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 0, 1, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 0, 0, 1), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 1, 0, 0, 1), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 1, 1, 0), output_path, (name + "{}.png".format(iterate())))
	save(img, output_path, (name + "{}.png".format(iterate())))


def augment_to_40(img, output_path, name, num_chanels):
	'''
	this saves the original image and 39 augmentations of it in the augmented path
	:param img: the image to augment and then save
	:param output_path: path to save images in
	:param name: Name of the output images
	:param num_chanels: to not add color grains to greyscale images.
	'''
	global i
	save(affine_transform(img, num_chanels), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.6), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.7), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.8), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 0.9), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.1), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.2), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.3), output_path, (name + "{}.png".format(iterate())))
	save(brighten_darken(img, 1.4), output_path, (name + "{}.png".format(iterate())))
	save(add_light_and_color_blur(img), output_path, (name + "{}.png".format(iterate())))
	save(blur(img), output_path, (name + "{}.png".format(iterate())))
	save(sharpen(img), output_path, (name + "{}.png".format(iterate())))

	if num_chanels == 1:
		save(add_salt_and_pepper_noise(mirror(img), 350), output_path, (name + "{}.png".format(iterate())))
		save(add_salt_and_pepper_noise(mirror(img), 150), output_path, (name + "{}.png".format(iterate())))
		save(add_salt_and_pepper_noise(mirror(img), 75), output_path, (name + "{}.png".format(iterate())))
		save(add_salt_and_pepper_noise(mirror(img), 40), output_path, (name + "{}.png".format(iterate())))
	else:
		save(add_tutti_frutty_noise(img, 400), output_path, (name + "{}.png".format(iterate())))
		save(add_tutti_frutty_noise(img, 200), output_path, (name + "{}.png".format(iterate())))
		save(add_tutti_frutty_noise(img, 100), output_path, (name + "{}.png".format(iterate())))
		save(add_tutti_frutty_noise(img, 50), output_path, (name + "{}.png".format(iterate())))

	save(pepper(img, 400), output_path, (name + "{}.png".format(iterate())))
	save(pepper(img, 200), output_path, (name + "{}.png".format(iterate())))
	save(pepper(img, 100), output_path, (name + "{}.png".format(iterate())))
	save(pepper(img, 50), output_path, (name + "{}.png".format(iterate())))
	save(salt(img, 400), output_path, (name + "{}.png".format(iterate())))
	save(salt(img, 200), output_path, (name + "{}.png".format(iterate())))
	save(salt(img, 100), output_path, (name + "{}.png".format(iterate())))
	save(salt(img, 50), output_path, (name + "{}.png".format(iterate())))
	save(add_salt_and_pepper_noise(img, 400), output_path, (name + "{}.png".format(iterate())))
	save(add_salt_and_pepper_noise(img, 200), output_path, (name + "{}.png".format(iterate())))
	save(add_salt_and_pepper_noise(img, 100), output_path, (name + "{}.png".format(iterate())))
	save(add_salt_and_pepper_noise(img, 50), output_path, (name + "{}.png".format(iterate())))
	save(mirror(img), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, -2), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, -1), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, 1), output_path, (name + "{}.png".format(iterate())))
	save(rotate(img, 2), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 1, 0, 0, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 1, 0, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 0, 1, 0), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 0, 0, 1), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 1, 0, 0, 1), output_path, (name + "{}.png".format(iterate())))
	save(crop(img, 0, 1, 1, 0), output_path, (name + "{}.png".format(iterate())))
	save(img, output_path, (name + "{}.png".format(iterate())))


def crop(img, left, right, top, bottom, shape=False):
	'''
	crops the images sides and reshapes the
	:param img: the image to crop
	:param left: how many pixels to crop off the left
	:param right: how many pixels to crop off the right
	:param top: how many pixels to crop off the top
	:param bottom: how many pixels to crop off the bottom
	:param shape: the new shape of the image, without this the original shape will be retained
	:return: the modified image
	'''
	# copies the image
	cpy = np.copy(img)

	# checks if the it should use the original shape or a new shape
	if shape==False:
		new_row = cpy.shape[0]
		new_col = cpy.shape[1]
	else:
		new_row = shape[0]
		new_col = shape[1]

	# gets the rows and cols to use for cropping
	rows = cpy.shape[1]
	cols = cpy.shape[0]

	# crops the image
	cropped = cpy[right:rows-left, top:cols-bottom]

	# resizes the image
	cpy = resize(cropped, (new_row, new_col))

	return cpy


def rotate(img, degree):
	'''
	rotates the image by x degrees
	:param img: the image to rotate
	:param degree: amount of degrees to rotate
	:return: the modified image
	'''
	# copies the image
	cpy = np.copy(img)

	# saves the rows and cols
	rows = cpy.shape[1]
	cols = cpy.shape[0]

	# generates a rotation matrix
	rotation = cv2.getRotationMatrix2D((cols/2,rows/2), degree, 1)

	# rotate the image
	cpy = cv2.warpAffine(cpy, rotation, (cols, rows))

	return cpy


def mirror(img):
	'''
	this mirrors the image
	:param img: the image to mirror
	:return: the mirrored image
	'''
	# creates a copy of the image
	cpy = np.copy(img)

	# mirrors and then returns the image
	return cv2.flip(cpy, 1)


def add_salt_and_pepper_noise(img, range):
	'''
	adds salt and pepper noise to the image. This is not true to a normal salt and pepper as it adds noise
	in a range from 0 to 255 rather then either 0 or 255
	:param img: the image to add noise to
	:param range: how often to add noise. Lower means more noise. 0 means every pixel becomes noise and
					100 means every 100th pixel becomes noise
	:return: returns the augmented image
	'''

	# creates a copy of the image
	cpy = np.copy(img)

	# saves the shape in cols and rows
	cols=cpy.shape[0]
	rows = cpy.shape[1]

	# initiates iteration variables
	i = 0
	j = 0

	# iterates through all the pixels in the image and changes its value if a random check gets met
	while i < rows:
		while j < cols:
			if rand.randrange(0,range) == 0:
				cpy[i][j] = rand.randrange(0,255)
			j = j + 1
		j=0
		i = i + 1

	return cpy


def salt(img, range):
	'''
	Adds salt noise to the image.
	:param img: image to add noise to
	:param range: how often to add noise. Lower means more noise. 0 means every pixel becomes noise and
					100 means every 100th pixel becomes noise
	:return: the augmented image
	'''

	# creates a copy of the image
	cpy = np.copy(img)

	# saves the shape in cols and rows
	cols=cpy.shape[0]
	rows = cpy.shape[1]

	# initiates iteration variables
	i = 0
	j = 0

	# iterates through all the pixels in the image and changes its value if a random check gets met
	while i < rows:
		while j < cols:
			if rand.randrange(0,range) == 0:
				cpy[i][j] = 255
			j = j + 1
		j=0
		i = i + 1

	return cpy


def pepper(img, range):
	'''
	adds pepper noise
	:param img: the image to add noise to
	:param range: how often to add noise. Lower means more noise. 0 means every pixel becomes noise and
					100 means every 100th pixel becomes noise
	:return: the modified image
	'''
	# creates a copy of the image
	cpy = np.copy(img)

	# saves the shape in cols and rows
	cols = cpy.shape[0]
	rows=cpy.shape[1]

	# initiates iteration variables
	i = 0
	j = 0

	# iterates through all the pixels in the image and changes its value if a random check gets met
	while i < rows:
		while j < cols:
			if rand.randrange(0,range) == 0:
				cpy[i][j] = 0
			j = j + 1
		j=0
		i = i + 1

	return cpy


def add_tutti_frutty_noise(img, range):
	'''
	Adds a similar noise to the salt and pepper noise however rather than being greyscale it is in color.
	:param img: the image to add noise to
	:param range: how often to add noise. Lower means more noise. 0 means every pixel becomes noise and
					100 means every 100th pixel becomes noise
	:return: the augmented image
	'''

	# creates a copy of the image
	cpy = np.copy(img)

	# saves the shape in cols and rows
	w=cpy.shape[1]
	h=cpy.shape[0]

	# initiates iteration variables
	i = 0
	j = 0

	# iterates through all the pixels in the image and changes its value if a random check gets met
	while i < w:
		while j < h:
			if rand.randrange(0,range) == 0:
				cpy[i][j][rand.randrange(0,2)] = rand.randrange(0,255)
			j = j + 1
		j=0
		i = i + 1

	return cpy


def sharpen(img):
	'''
	shapens the image
	:param img: the image to sharpen
	:return: returns the shaprened image
	'''

	#creates a copy of the image
	cpy = np.copy(img)

	# creates the shapening kernel
	kernel = np.array([
		[0,		-1,		0], 
		[-1,	5,		-1], 
		[0,		-1,		0]])

	# applies the kernel to the image
	cpy = cv2.filter2D(cpy, -1, kernel)

	return cpy


def blur(img):
	"""
	adds a gaussian blur to the image
	:param img: the image to blur
	:return: the blurred image
	"""
	# Copies the image
	cpy = np.copy(img)

	# creates the gaussian kernel
	kernel = np.array([
		[(1/9)*1,		(1/9)*1,		(1/9)*1], 
		[(1/9)*1,		(1/9)*1,		(1/9)*1], 
		[(1/9)*1,		(1/9)*1,		(1/9)*1]])

	# applies the kernel
	cpy = cv2.filter2D(cpy, -1, kernel)

	return cpy


def add_light_and_color_blur(img):
	"""
	Adds a blur and slightly brightens the image
	:param img: the image to modify
	:return:the augmented image
	"""

	# creates a copy of the image
	cpy = np.copy(img)

	# creates the blurring and brightening kernel
	kernel = np.array([
		[0.1,		0.1,		0.1], 
		[0.1,		1,		0.1], 
		[0.1,		0.1,		0.1]])

	# applies the kernel to the image
	cpy = cv2.filter2D(cpy, -1, kernel)

	return cpy


def brighten_darken(img, mult): #abve 1 to brighten, below to darken
	"""
	Brightens or darkens the image based on the multiplier
	:param img: the image to augment
	:param mult: abve 1 to brighten, below 1 to darken
	:return: returns the augmented image
	"""
	# copies the image
	cpy = np.copy(img)

	# creates the kernel
	kernel = np.array([mult])

	# applies the kernel to the image
	cpy = cv2.filter2D(cpy, -1, kernel)

	return cpy


def add_edge(img, edge, num_channels):
	"""
	adds an edge around the image, used for the affine transform
	:param img: the image to add an edge to
	:param edge: the size of the edge you want
	:param num_channels: number of channels
	:return: original image with an edge
	"""
	# creates a copy of the image
	cpy = np.copy(img)

	# saves the rows and colums
	rows = cpy.shape[1]
	cols = cpy.shape[0]

	# creates the new image size dependant on if it is a color image or not
	if num_channels == 1:
		base_size = cols + (edge * 2), rows + (edge * 2)
	else:
		base_size = cols + (edge * 2), rows + (edge * 2), num_channels

	# make an image for base which is larger than the original img
	base = np.zeros(base_size, dtype=np.uint8)

	# adds the original image to the base image
	base[edge:edge + cpy.shape[0], edge:edge + cpy.shape[1]] = cpy

	return base


def affine_transform(img, num_channels=3, original_points=np.float32([[5, 500], [20, 5], [5, 20]]), new_points=np.float32([[5, 498], [20, 5], [4, 20]])):
	"""
	Does the a affine transform on the input image. This means parallel lines stay parallel
	:param img: image to transform
	:param num_channels: redundant.
	:param original_points: a set of 3 points to be used as reference
	:param new_points: a set of tree new points that the original points gets moved to
	:return: transformed image
	"""
	#copies the image
	cpy = np.copy(img)

	# saves the rows and colums
	rows = cpy.shape[1]
	cols = cpy.shape[0]

	# creates the rotation matrix
	rotation = cv2.getAffineTransform(original_points, new_points)

	# applies the rotation matrix and transforms the image
	dst = cv2.warpAffine(cpy,rotation,(cols,rows))

	return dst

