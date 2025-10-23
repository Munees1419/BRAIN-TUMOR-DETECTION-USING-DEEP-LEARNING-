import numpy as np 
from tqdm import tqdm
import cv2
import os as ops
import shutil
import itertools
import random
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import preprocess as pre
 

#pip install numpy tqdm opencv-python imutils matplotlib scikit-learn plotly tensorflow scikit-image

#python -m venv venv     venv\Scripts\activate


init_notebook_mode(connected=True)
RANDOM_SEED = 123

IMG_PATH="C:\munees\img14"

IMG_SIZE = (224,224)

TRAIN_DIR = 'C:/munees/TRAIN'
TEST_DIR  = 'C:/munees/TEST'
VAL_DIR   = 'C:/munees/VAL'

# split the data by train/val/test
for CLASS in ops.listdir(IMG_PATH):
    if not CLASS.startswith('.'):
        class_path = ops.path.join(IMG_PATH, CLASS)
        IMG_NUM = len(ops.listdir(class_path))

        for n, FILE_NAME in enumerate(ops.listdir(class_path)):
            img = ops.path.join(class_path, FILE_NAME)

            # TEST (first 5 images)
            if n < 5:
                dst_dir = ops.path.join(TEST_DIR, CLASS.upper())
            
            # TRAIN (next 80%)
            elif n < 0.8 * IMG_NUM:
                dst_dir = ops.path.join(TRAIN_DIR, CLASS.upper())

            # VAL (remaining images)
            else:
                dst_dir = ops.path.join(VAL_DIR, CLASS.upper())

            # ✅ Create the folder if it doesn't exist
            ops.makedirs(dst_dir, exist_ok=True)

            # ✅ Copy the image safely
            shutil.copy(img, ops.path.join(dst_dir, FILE_NAME))

# use predefined function to load the image data into workspace
X_train, y_train, labels = pre.load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = pre.load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = pre.load_data(VAL_DIR, IMG_SIZE)

y = dict()
y[0] = []
y[1] = []
for set_name in (y_train, y_val, y_test):
    y[0].append(np.sum(set_name == 0))
    y[1].append(np.sum(set_name == 1))

trace0 = go.Bar(
    x=['Train Set', 'Validation Set', 'Test Set'],
    y=y[0],
    name='No',
    marker=dict(color='#33cc33'),
    opacity=0.7
)
trace1 = go.Bar(
    x=['Train Set', 'Validation Set', 'Test Set'],
    y=y[1],
    name='Yes',
    marker=dict(color='#ff3300'),
    opacity=0.7
)
data = [trace0, trace1]
layout = go.Layout(
    title='Count of classes in each set',
    xaxis={'title': 'Set'},
    yaxis={'title': 'Count'}
)
fig = go.Figure(data, layout)
iplot(fig)

#pre.plot_samples(X_train, y_train, labels, 30)

RATIO_LIST = []
for set in (X_train, X_test, X_val):
    for img in set:
        RATIO_LIST.append(img.shape[1]/img.shape[0])
        
plt.hist(RATIO_LIST)
plt.title('Distribution of Image Ratios')
plt.xlabel('Ratio Value')
plt.ylabel('Count')
plt.show()

img = cv2.imread(r"C:\munees\testing images\tumor\yes_001.png")
img = cv2.resize(img,dsize=IMG_SIZE,interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# find the extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# add contour on the image
img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

# add extreme points
img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

# crop
ADD_PIXELS = 0
new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

plt.figure(figsize=(15,6))
plt.subplot(141)
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title('Step 1. Get the original image')
plt.subplot(142)
plt.imshow(img_cnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 2. Find the biggest contour')
plt.subplot(143)
plt.imshow(img_pnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 3. Find the extreme points')
plt.subplot(144)
plt.imshow(new_img)
plt.xticks([])
plt.yticks([])
plt.title('Step 4. Crop the image')
plt.show()

# apply this for each set
X_train_crop = pre.crop_imgs(set_name=X_train)
X_val_crop = pre.crop_imgs(set_name=X_val)
X_test_crop =pre.crop_imgs(set_name=X_test)

#pre.plot_samples(X_train_crop, y_train, labels, 30)

pre.save_new_images(X_train_crop, y_train, folder_name='TRAIN_CROP/')
pre.save_new_images(X_val_crop, y_val, folder_name='VAL_CROP/')
pre.save_new_images(X_test_crop, y_test, folder_name='TEST_CROP/')

X_train_prep = pre.preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
X_test_prep = pre.preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
X_val_prep = pre.preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)

pre.plot_samples(X_train_prep, y_train, labels, 30)


demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

if not ops.path.exists('preview'):
    ops.mkdir('preview')

x = X_train_crop[0]  
x = x.reshape((1,) + x.shape) 

i = 0
for batch in demo_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='aug_img', save_format='jpg'):
    i += 1
    if i > 20:
        break 

plt.imshow(X_train_crop[0])
plt.xticks([])
plt.yticks([])
plt.title('Original Image')
plt.show()

plt.figure(figsize=(15,6))
i = 1
for img in ops.listdir('preview/'):
    img = cv2.imread('preview/' + img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3,7,i)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    i += 1
    if i > 3*7:
        break
plt.suptitle('Augemented Images')
plt.show()

TRAIN_DIR = 'C:\munees\TRAIN_CROP'
VAL_DIR = 'C:\munees\VAL_CROP'

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR ,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=RANDOM_SEED
)


validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)

vgg16_weight_path = 'C:\munees\trained model\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = VGG16(
    weights='imagenet',
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)

model.summary()

train_samples = train_generator.samples
val_samples = validation_generator.samples
batch_size = train_generator.batch_size  # use same batch size for val_generator

#steps_per_epochs = train_samples // batch_size
#validation_step = val_samples // batch_size

steps_per_epochs = int(np.ceil(train_samples / batch_size))
validation_step = int(np.ceil(val_samples / batch_size))



EPOCHS = 30
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
es = EarlyStopping(
    monitor='val_accuracy', 
    mode='min',
    patience=5
)

history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epochs,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=validation_step,
                    callbacks=[es,reduce_lr])

# Plot model performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predict segmentation masks
val_preds = model.predict(X_val_prep)
test_preds = model.predict(X_test_prep)

# Convert probabilities to binary masks
val_preds_bin = (val_preds > 0.5).astype(np.uint8)
test_preds_bin = (test_preds > 0.5).astype(np.uint8)

# Pixel-wise accuracy (overall)
val_pixel_acc = np.mean(val_preds_bin == y_val)
test_pixel_acc = np.mean(test_preds_bin == y_test)

print(f'✅ Validation Pixel Accuracy = {val_pixel_acc:.2f}')
print(f'✅ Test Pixel Accuracy = {test_pixel_acc:.2f}')

for i in range(len(X_test_prep)):
    true_mask = y_test[i]
    pred_mask = test_preds_bin[i]

    # Check if tumor is present in the ground truth
    if np.sum(true_mask) == 0:
        # No tumor in ground truth — skip visualization
        continue

    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(X_test_prep[i])
    plt.title("Original Image")
    plt.axis('off')

    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.squeeze(), cmap='gray')  # Use squeeze safely
    plt.title("Ground Truth Mask")
    plt.axis('off')

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), cmap='gray')  # Use squeeze safely
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


#------------------
# save the model
#------------------
model.save('MRI1419_VGG_model.h5')

