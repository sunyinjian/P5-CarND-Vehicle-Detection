from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time
import pickle
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# define a function to return HOG feature and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis,
                                  feature_vector=feature_vec)
        return features


# define a function to extract features from a list of images
def extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


car_paths = glob('./training_data/vehicles/*/*.png')
noncar_paths = glob('./training_data/non-vehicles/*/*.png')


# visualize some examples of data
print('Num of cars: {}'.format(len(car_paths)))
print('Num of noncars: {}'.format(len(noncar_paths)))
print('Image shape: {}'.format(mpimg.imread(car_paths[0]).shape))

f, ax = plt.subplots(6, 6, figsize = (10, 10))
f.subplots_adjust(hspace=.3, wspace=.1)
ax = ax.ravel()
for i in range(18):
    ind = np.random.randint(0, len(car_paths))
    car_img = mpimg.imread(car_paths[ind])
    ax[i].axis('off')
    ax[i].set_title('car', fontsize=10)
    ax[i].imshow(car_img)
for i in range(18,36):
    ind = np.random.randint(0, len(noncar_paths))
    noncar_img = mpimg.imread(noncar_paths[ind])
    ax[i].axis('off')
    ax[i].set_title('noncar', fontsize=10)
    ax[i].imshow(noncar_img)


# visualize the HOG feature on an example image
ind = np.random.randint(0, len(car_paths))
car_img = mpimg.imread(car_paths[ind])
_, car_hog = get_hog_features(car_img[:,:,2] , 9, 8, 2, vis=True, feature_vec=False)

ind = np.random.randint(0, len(noncar_paths))
noncar_img = mpimg.imread(noncar_paths[ind])
_, noncar_hog = get_hog_features(noncar_img[:,:,2] , 9, 8, 2, vis=True, feature_vec=False)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5,5))
f.tight_layout()
ax1.set_title('Car Img', fontsize=10)
ax1.imshow(car_img)
ax2.set_title('Car HOG', fontsize=10)
ax2.imshow(car_hog, cmap='gray')

ax3.imshow(noncar_img)
ax3.set_title('Noncar Img', fontsize=10)
ax4.imshow(noncar_hog, cmap='gray')
ax4.set_title('Noncar HOG', fontsize=10)


# set parameters for feature extraction
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"


t = time.time()
car_features = extract_features(car_paths, cspace=color_space, orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block, hog_channel=hog_channel)
noncar_features = extract_features(noncar_paths, cspace=color_space, orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block, hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
features = np.vstack((car_features, noncar_features)).astype(np.float64)

# fit a per-column scaler
X_scaler = StandardScaler().fit(features)
# apply the scaler to X
scaled_X = X_scaler.transform(features)

#set labels
y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

#split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

# Train a classifier
# use a linear svc
svc = LinearSVC()
t = time.time()

svc.fit(X_train, y_train)
t1 = time.time()

print('Time consumed for training: {} sec'.format(round(t1-t, 2)))
print('Test Accuracy of SVC = {}'.format(round(svc.score(X_test, y_test), 4)))

#save the model and corresponding parameters
model = {}
model['svc'] = svc
model['X_scaler'] = X_scaler
model['color_space'] = color_space
model['orient'] = orient
model['pix_per_cell'] = pix_per_cell
model['cell_per_block'] = cell_per_block
model['hog_channel'] = hog_channel
pickle.dump(model, open('model.p', 'wb'))

plt.show()
