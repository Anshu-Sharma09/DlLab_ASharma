import numpy as np

# ==================== CREATE INPUT IMAGE ====================

image=np.random.randint(0,256,(32,32))

# np.random.randint(0,256) generates random pixel values from 0 to 255.
# Why 256?
# A normal 8-bit image has 256 possible intensity values:
# 0 = black
# 255 = white
# Therefore the range is 0 to 255, and Python uses 256 as the upper limit.
#
# (32,32) means:
# 32 rows of pixels
# 32 columns of pixels
# Therefore input image size = 32 x 32


# ==================== CREATE 3x3 KERNEL ====================

kernel=np.array([[1,0,-1],
                 [1,0,-1],
                 [1,0,-1]])
# Different kernels detect different features such as edges,
# sharpening, blurring, etc.


# ==================== CONVOLUTION ====================

# Formula for convolution output size:
#
# Output = ((Input size - Kernel size + 2*Padding) / Stride) + 1

#
# Therefore convolution output = 30 x 30

conv_output=np.zeros((30,30))

# np.zeros((30,30)) creates an empty 30x30 matrix
# to store the convolution results.


# i and j represent the position of the kernel on the image.
# We need 30 positions in both directions because the output is 30x30.
for i in range(30):
    for j in range(30):

        # Take a 3x3 region from the 32x32 image.
        #
        # i:i+3 means take 3 rows
        # j:j+3 means take 3 columns
        #
        # Example:
        # image[0:3,0:3] gives the first 3x3 region.
        #
        # The 3x3 kernel is then applied to this 3x3 region.
        region=image[i:i+3,j:j+3]

        # Multiply corresponding values of the image region
        # and kernel, then add all the values.
        #
        # This produces ONE value in the convolution output.
        conv_output[i,j]=np.sum(region*kernel)


print("Input shape:",image.shape)
print("Kernel shape:",kernel.shape)
print("Convolution output shape:",conv_output.shape)


# ==================== MAX POOLING ====================
pool_size=2
stride=2


# Calculate max-pooling output size:
#
# Input = 30
# Pool size = 2
# Padding = 0
# Stride = 2
#
# Output = ((30 - 2 + 2*0) / 2) + 1
#        = (28 / 2) + 1
#        = 14 + 1
#        = 15
#
# Therefore output = 15 x 15

pool_output=np.zeros((15,15))


# There are 15 positions in each direction,
# so we use range(15).
for i in range(15):
    for j in range(15):

        # Select a 2x2 region from the convolution output.
        region=conv_output[i*stride:i*stride+pool_size,
                           j*stride:j*stride+pool_size]

        # np.max() selects the largest value from the 2x2 region.
        # This is the main operation of MAX POOLING.
        pool_output[i,j]=np.max(region)


print("MaxPool output shape:",pool_output.shape)
print("MaxPool output:")
print(pool_output)