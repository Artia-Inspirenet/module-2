###############################################################################
#                  EDGE DETECTION 3.1 Canny Edge Detection                    #
#                                By: Todd Farr                                #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt





################################################ CANNY EDGE DETECTION EXAMPLE 2
# The Lena image is a really common image for Canny edge Detection
# read in the image
lena = cv2.imread('lena.png')

# convert image to grayscale (NOTE: we could have used 0 as arg 2 for imread())
lena_grayscale = cv2.cvtColor(lena, cv2.COLOR_RGB2GRAY)

# find canny edges
lena_canny = cv2.Canny(lena_grayscale, 100, 200)
cv2.imshow('Lena Canny Edge Detection', lena_canny)
cv2.waitKey()
cv2.destroyAllWindows()


# #################################################### EFFECTS OF LOWER THRESHOLD
# # Exploring the effects of the lower threshold value with Canny Detection
# fig = plt.figure(figsize=(20, 10))
# fig.canvas.set_window_title(
#     'The Effects of the Lower Threshold Value on Canny Edge Detection')
# for i, value in enumerate(range(10, 181, 10)):
#     canny = cv2.Canny(lena_grayscale, value, 200)
#     plt.subplot(3, 6, i + 1), plt.title('Lower Threshold = {}'.format(value))
#     plt.imshow(canny, cmap='gray'), plt.xticks([]), plt.yticks([])

# plt.tight_layout()
# plt.show()


# #################################################### EFFECTS OF UPPER THRESHOLD
# # Exploring the effects of the upper threshold value with Canny Detection
# fig = plt.figure(figsize=(20, 10))
# fig.canvas.set_window_title(
#     'The Effects of the Upper Threshold Value Canny Edge Detection')
# for i, value in enumerate(range(210, 381, 10)):
#     canny = cv2.Canny(lena_grayscale, 100, value)
#     plt.subplot(3, 6, i + 1), plt.title('Upper Threshold = {}'.format(value))
#     plt.imshow(canny, cmap='gray'), plt.xticks([]), plt.yticks([])

# plt.tight_layout()
# plt.show()


# ####################################################### EFFECTS OF APATURE SIZE
# # Exploring the effects of apature size with Canny Edge Detection
# # Remember this is this aperture size for computing the gradients in the Sobel
# # opeator. Therefore in OpenCV Sobel(), this would be ksize and so our only
# # choices indicated by the documentation are 3, 5 & 7
# fig = plt.figure(figsize=(10, 10))
# fig.canvas.set_window_title(
#     'The Effects of the Aperture Size Canny Edge Detection')
# for i, value in enumerate(range(3, 8, 2)):
#     canny = cv2.Canny(lena_grayscale, 100, 200, apertureSize = value)
#     plt.subplot(2, 2, i + 1), plt.title('Aperture Size = {}'.format(value))
#     plt.imshow(canny, cmap='gray'), plt.xticks([]), plt.yticks([])

# plt.tight_layout()
# plt.show()


# ############################################################## EFFECTS OF SIGMA
# # Unfortunately, the OpenCV Canny function doesn't let you change the filter
# # kernel it uses via the function parameters. However. You can generate the
# # same results by first blurring the input image, and then passing this blurred
# # image into the Canny function.
# #
# # Since both the Gaussian Blur and Sobel filters are linear, passing a blurred
# # input image to the OpenCV Canny() function is mathematically equivalent to
# # what Matlab does because of the principle of superposition.
# # (NOTE: *This assumes this is the convolution operator)
# #
# # The Matlab method: the sobel and blur operations are combined into
# # a single filter, and that filter is then convolved with the image
# #### matlabFancyFilter = (sobel * blur);
# #### gradient = matlabFancyFilter * image;
# #
# # Equivalent method: image is first convolved with the blur filter, and
# # then convolved with the sobel filter.
# #### gradient = sobel * (blur * image); // image is filtered twice


# # Testing Sigma values between 0.5 - 2 with a constant filter size (7 x 7)
# fig = plt.figure(figsize=(10, 10))
# fig.canvas.set_window_title(
#     'The Effects of the Sigma (GaussianBlur) on Canny Edge Detection')
# for i, value in enumerate(range(1, 5)):
#     smoothed_lena = cv2.GaussianBlur(lena_grayscale, (7, 7), value / 2.)
#     canny = cv2.Canny(smoothed_lena, 100, 200)
#     plt.subplot(2, 2, i + 1), plt.title('Sigma = {:.1f}'.format(value / 2.))
#     plt.imshow(canny, cmap='gray'), plt.xticks([]), plt.yticks([])

# plt.tight_layout()
# plt.show()
