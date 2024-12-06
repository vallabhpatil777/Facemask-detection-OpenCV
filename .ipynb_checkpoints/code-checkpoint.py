{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "db0da39f-abc0-46e5-a90c-fd13c5cbaf69",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from keras.preprocessing import image\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "90a34f41-6166-4a5b-b0b1-b8c6539f7d3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "categories = ['with_mask','without_mask']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ead5e85d-997c-4bdf-bc0a-702023255cdb",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = []\n",
    "for category in categories:\n",
    "    path = os.path.join('dataset',category)\n",
    "    label = categories.index(category)\n",
    "    for file in os.listdir(path):\n",
    "        imgpath = os.path.join(path,file)\n",
    "        img = cv2.imread(imgpath)\n",
    "        img = cv2.resize(img,(224,224))\n",
    "        data.append([img,label])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1a828e32-f28a-42c3-80be-18aaf3b3d1ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7553"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bde4d3f9-d98a-460b-80a1-d2a1f74800a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "81b06849-f007-4d54-8c8e-9f32ac5a7fbf",
   "metadata": {},
   "outputs": [],
   "source": [
    "random.shuffle(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "87db5cbd-064c-43b3-b259-d37c08755ea8",
   "metadata": {},
   "outputs": [],
   "source": [
    "X= []\n",
    "y=[]\n",
    "for features,label in data:\n",
    "    X.append(features)\n",
    "    y.append(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "73095c29-d537-4ebe-9334-35ac25f8bdea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7553"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "69ec3355-4cae-4d66-af86-1fa9ebd73049",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7553"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f842a779-543d-4640-b53d-832ae0882945",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "76b6fe54-1e82-4137-b185-e6257579902a",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = np.array(X)\n",
    "y = np.array(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "fe6b1004-dd2c-4b10-973a-1e8486fb4c32",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7553, 224, 224, 3)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e30f5a25-0b04-4aa0-844c-71719cb28eaf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7553,)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "0b62f405-b6ab-47e8-acd2-484e21a59b98",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = X/255\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "12bcacb9-5885-4ba2-9d60-ccf839728444",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6042, 224, 224, 3)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "23f48a63-678c-46fd-bd11-1ade0aaf307b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1511, 224, 224, 3)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "2190733b-57e1-43d2-82d8-220c060593b4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"vgg16\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"vgg16\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ input_layer (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">InputLayer</span>)        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │         <span style=\"color: #00af00; text-decoration-color: #00af00\">1,792</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │        <span style=\"color: #00af00; text-decoration-color: #00af00\">36,928</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)  │        <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)  │       <span style=\"color: #00af00; text-decoration-color: #00af00\">147,584</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">295,168</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">1,180,160</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">25088</span>)          │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │   <span style=\"color: #00af00; text-decoration-color: #00af00\">102,764,544</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │    <span style=\"color: #00af00; text-decoration-color: #00af00\">16,781,312</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ predictions (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1000</span>)           │     <span style=\"color: #00af00; text-decoration-color: #00af00\">4,097,000</span> │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ input_layer (\u001b[38;5;33mInputLayer\u001b[0m)        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m3\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │         \u001b[38;5;34m1,792\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │        \u001b[38;5;34m36,928\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m128\u001b[0m)  │        \u001b[38;5;34m73,856\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m128\u001b[0m)  │       \u001b[38;5;34m147,584\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m295,168\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m1,180,160\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m25088\u001b[0m)          │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc1 (\u001b[38;5;33mDense\u001b[0m)                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │   \u001b[38;5;34m102,764,544\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc2 (\u001b[38;5;33mDense\u001b[0m)                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │    \u001b[38;5;34m16,781,312\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ predictions (\u001b[38;5;33mDense\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1000\u001b[0m)           │     \u001b[38;5;34m4,097,000\u001b[0m │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">138,357,544</span> (527.79 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m138,357,544\u001b[0m (527.79 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">138,357,544</span> (527.79 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m138,357,544\u001b[0m (527.79 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from keras.applications.vgg16 import VGG16\n",
    "vgg = VGG16()\n",
    "vgg.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "801d9f1c-0295-4420-8eb4-8deaee0107da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ block1_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │         <span style=\"color: #00af00; text-decoration-color: #00af00\">1,792</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │        <span style=\"color: #00af00; text-decoration-color: #00af00\">36,928</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)  │        <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)  │       <span style=\"color: #00af00; text-decoration-color: #00af00\">147,584</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">295,168</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">1,180,160</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">25088</span>)          │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │   <span style=\"color: #00af00; text-decoration-color: #00af00\">102,764,544</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │    <span style=\"color: #00af00; text-decoration-color: #00af00\">16,781,312</span> │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ block1_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │         \u001b[38;5;34m1,792\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │        \u001b[38;5;34m36,928\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m128\u001b[0m)  │        \u001b[38;5;34m73,856\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m128\u001b[0m)  │       \u001b[38;5;34m147,584\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m295,168\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m1,180,160\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m25088\u001b[0m)          │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc1 (\u001b[38;5;33mDense\u001b[0m)                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │   \u001b[38;5;34m102,764,544\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc2 (\u001b[38;5;33mDense\u001b[0m)                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │    \u001b[38;5;34m16,781,312\u001b[0m │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">134,260,544</span> (512.16 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m134,260,544\u001b[0m (512.16 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">134,260,544</span> (512.16 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m134,260,544\u001b[0m (512.16 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from keras import Sequential\n",
    "from keras.layers import Dense, Flatten, Dropout\n",
    "model = Sequential()\n",
    "\n",
    "for layer in vgg.layers[:-1]:\n",
    "    model.add(layer)\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1ebb965f-62a1-4bd2-8f86-43150c2eaaba",
   "metadata": {},
   "outputs": [],
   "source": [
    "for layer in model.layers:\n",
    "    layer.trainable = False\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "d9bddd07-f35c-4f5c-adf1-a9daa4aa7b76",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ block1_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │         <span style=\"color: #00af00; text-decoration-color: #00af00\">1,792</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │        <span style=\"color: #00af00; text-decoration-color: #00af00\">36,928</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)  │        <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)  │       <span style=\"color: #00af00; text-decoration-color: #00af00\">147,584</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">295,168</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">1,180,160</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">25088</span>)          │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │   <span style=\"color: #00af00; text-decoration-color: #00af00\">102,764,544</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │    <span style=\"color: #00af00; text-decoration-color: #00af00\">16,781,312</span> │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ block1_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │         \u001b[38;5;34m1,792\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │        \u001b[38;5;34m36,928\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m128\u001b[0m)  │        \u001b[38;5;34m73,856\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m128\u001b[0m)  │       \u001b[38;5;34m147,584\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m295,168\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m1,180,160\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m25088\u001b[0m)          │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc1 (\u001b[38;5;33mDense\u001b[0m)                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │   \u001b[38;5;34m102,764,544\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc2 (\u001b[38;5;33mDense\u001b[0m)                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │    \u001b[38;5;34m16,781,312\u001b[0m │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">134,260,544</span> (512.16 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m134,260,544\u001b[0m (512.16 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">134,260,544</span> (512.16 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m134,260,544\u001b[0m (512.16 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3ced7dcc-d435-4248-bf6b-91da7e759e41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ block1_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │         <span style=\"color: #00af00; text-decoration-color: #00af00\">1,792</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │        <span style=\"color: #00af00; text-decoration-color: #00af00\">36,928</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)   │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)  │        <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)  │       <span style=\"color: #00af00; text-decoration-color: #00af00\">147,584</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">295,168</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">56</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">1,180,160</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_pool (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">25088</span>)          │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │   <span style=\"color: #00af00; text-decoration-color: #00af00\">102,764,544</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │    <span style=\"color: #00af00; text-decoration-color: #00af00\">16,781,312</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>)              │         <span style=\"color: #00af00; text-decoration-color: #00af00\">4,097</span> │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ block1_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │         \u001b[38;5;34m1,792\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │        \u001b[38;5;34m36,928\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block1_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m64\u001b[0m)   │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m128\u001b[0m)  │        \u001b[38;5;34m73,856\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m112\u001b[0m, \u001b[38;5;34m128\u001b[0m)  │       \u001b[38;5;34m147,584\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block2_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m295,168\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m56\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block3_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m1,180,160\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block4_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv1 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv2 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_conv3 (\u001b[38;5;33mConv2D\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m512\u001b[0m)    │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ block5_pool (\u001b[38;5;33mMaxPooling2D\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m25088\u001b[0m)          │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc1 (\u001b[38;5;33mDense\u001b[0m)                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │   \u001b[38;5;34m102,764,544\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ fc2 (\u001b[38;5;33mDense\u001b[0m)                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │    \u001b[38;5;34m16,781,312\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1\u001b[0m)              │         \u001b[38;5;34m4,097\u001b[0m │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">134,264,641</span> (512.18 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m134,264,641\u001b[0m (512.18 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">4,097</span> (16.00 KB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m4,097\u001b[0m (16.00 KB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">134,260,544</span> (512.16 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m134,260,544\u001b[0m (512.16 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from keras.layers import Dense\n",
    "model.add(Dense(1, activation='sigmoid'))\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "7f6c54d7-39f4-4a5b-9172-3694402cc3d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.optimizers import Adam  \n",
    "\n",
    "model.compile(optimizer=Adam(learning_rate=1e-4),loss='binary_crossentropy',metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "d48b9af4-ccc6-4922-b585-e70ef448a110",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m392s\u001b[0m 2s/step - accuracy: 0.6978 - loss: 0.6274 - val_accuracy: 0.8206 - val_loss: 0.5164\n",
      "Epoch 2/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m380s\u001b[0m 2s/step - accuracy: 0.8416 - loss: 0.4890 - val_accuracy: 0.8676 - val_loss: 0.4335\n",
      "Epoch 3/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m380s\u001b[0m 2s/step - accuracy: 0.8570 - loss: 0.4214 - val_accuracy: 0.8776 - val_loss: 0.3837\n",
      "Epoch 4/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m380s\u001b[0m 2s/step - accuracy: 0.8758 - loss: 0.3752 - val_accuracy: 0.8882 - val_loss: 0.3503\n",
      "Epoch 5/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m381s\u001b[0m 2s/step - accuracy: 0.8948 - loss: 0.3376 - val_accuracy: 0.8961 - val_loss: 0.3293\n",
      "Epoch 6/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m380s\u001b[0m 2s/step - accuracy: 0.8925 - loss: 0.3226 - val_accuracy: 0.9093 - val_loss: 0.3056\n",
      "Epoch 7/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m382s\u001b[0m 2s/step - accuracy: 0.9001 - loss: 0.2962 - val_accuracy: 0.9014 - val_loss: 0.2984\n",
      "Epoch 8/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m383s\u001b[0m 2s/step - accuracy: 0.9060 - loss: 0.2823 - val_accuracy: 0.9186 - val_loss: 0.2770\n",
      "Epoch 9/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m379s\u001b[0m 2s/step - accuracy: 0.9050 - loss: 0.2763 - val_accuracy: 0.9199 - val_loss: 0.2655\n",
      "Epoch 10/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m382s\u001b[0m 2s/step - accuracy: 0.9113 - loss: 0.2640 - val_accuracy: 0.9186 - val_loss: 0.2610\n",
      "Epoch 11/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m378s\u001b[0m 2s/step - accuracy: 0.9127 - loss: 0.2589 - val_accuracy: 0.9272 - val_loss: 0.2476\n",
      "Epoch 12/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m378s\u001b[0m 2s/step - accuracy: 0.9165 - loss: 0.2402 - val_accuracy: 0.9272 - val_loss: 0.2393\n",
      "Epoch 13/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m378s\u001b[0m 2s/step - accuracy: 0.9251 - loss: 0.2297 - val_accuracy: 0.9325 - val_loss: 0.2327\n",
      "Epoch 14/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m380s\u001b[0m 2s/step - accuracy: 0.9202 - loss: 0.2263 - val_accuracy: 0.9365 - val_loss: 0.2266\n",
      "Epoch 15/15\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m380s\u001b[0m 2s/step - accuracy: 0.9227 - loss: 0.2166 - val_accuracy: 0.9351 - val_loss: 0.2211\n"
     ]
    }
   ],
   "source": [
    "\n",
    "history = model.fit(\n",
    "    X_train,y_train,epochs=15,validation_data=(X_test,y_test)\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "3245db9f-427f-4cfc-af69-2afb8f426a85",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved to model.pkl\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "\n",
    "# Assuming 'model' is your trained model\n",
    "model_file = \"model.pkl\"\n",
    "\n",
    "# Save the model\n",
    "with open(model_file, 'wb') as file:\n",
    "    pickle.dump(model, file)\n",
    "\n",
    "print(f\"Model saved to {model_file}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "a6ca10f8-1c15-4061-99f0-f6c121d13e01",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved as 'model.h5'\n"
     ]
    }
   ],
   "source": [
    "# Save the trained model\n",
    "model.save('model.h5')\n",
    "print(\"Model saved as 'model.h5'\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8a99d50e-61f2-429b-96b2-2ef67cc1ab9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model loaded successfully!\n"
     ]
    }
   ],
   "source": [
    "from keras.models import load_model\n",
    "\n",
    "# Load the model\n",
    "model = load_model('model.h5')\n",
    "print(\"Model loaded successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8b05c67f-ad69-454d-8c6c-4f9d2eea0f8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "3b8d4352-e0a0-41b9-b8f4-1ca2bf023f74",
   "metadata": {},
   "outputs": [],
   "source": [
    "def detect_face_mask(img):\n",
    "    y_pred = model.predict(img.reshape(1,224,224,3))\n",
    "    return y_pred[0][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cf1ad8c1-29d6-4086-9102-41278ef4bb38",
   "metadata": {},
   "outputs": [],
   "source": [
    "def draw_label(img,text,pos,bg_color):\n",
    "    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)\n",
    "    end_x = pos[0] + text_size[0][0] + 2\n",
    "    end_y = pos[1] + text_size[0][1] - 2\n",
    "    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)\n",
    "    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "16fe2e3b-1ce1-4627-aef2-f94485736fbc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 93ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 91ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 87ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 88ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 87ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 88ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 74ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 74ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 87ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 89ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 97ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 88ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 87ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 87ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 76ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 77ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 75ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 85ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 74ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 78ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 87ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 89ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 80ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 81ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 79ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 86ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 87ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 87ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 84ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 83ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[13], line 11\u001b[0m\n\u001b[1;32m      9\u001b[0m img \u001b[38;5;241m=\u001b[39m cv2\u001b[38;5;241m.\u001b[39mresize(frame, (\u001b[38;5;241m224\u001b[39m, \u001b[38;5;241m224\u001b[39m))\n\u001b[1;32m     10\u001b[0m img \u001b[38;5;241m=\u001b[39m img \u001b[38;5;241m/\u001b[39m \u001b[38;5;241m255.0\u001b[39m\n\u001b[0;32m---> 11\u001b[0m y_pred \u001b[38;5;241m=\u001b[39m detect_face_mask(img)\n\u001b[1;32m     13\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m y_pred \u001b[38;5;241m<\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0.10\u001b[39m:\n\u001b[1;32m     14\u001b[0m     draw_label(frame, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mMask\u001b[39m\u001b[38;5;124m\"\u001b[39m, (\u001b[38;5;241m30\u001b[39m, \u001b[38;5;241m30\u001b[39m), (\u001b[38;5;241m0\u001b[39m, \u001b[38;5;241m255\u001b[39m, \u001b[38;5;241m0\u001b[39m))\n",
      "Cell \u001b[0;32mIn[8], line 2\u001b[0m, in \u001b[0;36mdetect_face_mask\u001b[0;34m(img)\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mdetect_face_mask\u001b[39m(img):\n\u001b[0;32m----> 2\u001b[0m     y_pred \u001b[38;5;241m=\u001b[39m model\u001b[38;5;241m.\u001b[39mpredict(img\u001b[38;5;241m.\u001b[39mreshape(\u001b[38;5;241m1\u001b[39m,\u001b[38;5;241m224\u001b[39m,\u001b[38;5;241m224\u001b[39m,\u001b[38;5;241m3\u001b[39m))\n\u001b[1;32m      3\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m y_pred[\u001b[38;5;241m0\u001b[39m][\u001b[38;5;241m0\u001b[39m]\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py:117\u001b[0m, in \u001b[0;36mfilter_traceback.<locals>.error_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    115\u001b[0m filtered_tb \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    116\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 117\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m fn(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[1;32m    118\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[1;32m    119\u001b[0m     filtered_tb \u001b[38;5;241m=\u001b[39m _process_traceback_frames(e\u001b[38;5;241m.\u001b[39m__traceback__)\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/keras/src/backend/tensorflow/trainer.py:504\u001b[0m, in \u001b[0;36mTensorFlowTrainer.predict\u001b[0;34m(self, x, batch_size, verbose, steps, callbacks)\u001b[0m\n\u001b[1;32m    502\u001b[0m callbacks\u001b[38;5;241m.\u001b[39mon_predict_batch_begin(step)\n\u001b[1;32m    503\u001b[0m data \u001b[38;5;241m=\u001b[39m get_data(iterator)\n\u001b[0;32m--> 504\u001b[0m batch_outputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mpredict_function(data)\n\u001b[1;32m    505\u001b[0m outputs \u001b[38;5;241m=\u001b[39m append_to_outputs(batch_outputs, outputs)\n\u001b[1;32m    506\u001b[0m callbacks\u001b[38;5;241m.\u001b[39mon_predict_batch_end(step, {\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124moutputs\u001b[39m\u001b[38;5;124m\"\u001b[39m: batch_outputs})\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/util/traceback_utils.py:150\u001b[0m, in \u001b[0;36mfilter_traceback.<locals>.error_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    148\u001b[0m filtered_tb \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    149\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 150\u001b[0m   \u001b[38;5;28;01mreturn\u001b[39;00m fn(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[1;32m    151\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[1;32m    152\u001b[0m   filtered_tb \u001b[38;5;241m=\u001b[39m _process_traceback_frames(e\u001b[38;5;241m.\u001b[39m__traceback__)\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:833\u001b[0m, in \u001b[0;36mFunction.__call__\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    830\u001b[0m compiler \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mxla\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_jit_compile \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnonXla\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    832\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m OptionalXlaContext(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_jit_compile):\n\u001b[0;32m--> 833\u001b[0m   result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_call(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwds)\n\u001b[1;32m    835\u001b[0m new_tracing_count \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mexperimental_get_tracing_count()\n\u001b[1;32m    836\u001b[0m without_tracing \u001b[38;5;241m=\u001b[39m (tracing_count \u001b[38;5;241m==\u001b[39m new_tracing_count)\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:878\u001b[0m, in \u001b[0;36mFunction._call\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    875\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_lock\u001b[38;5;241m.\u001b[39mrelease()\n\u001b[1;32m    876\u001b[0m \u001b[38;5;66;03m# In this case we have not created variables on the first call. So we can\u001b[39;00m\n\u001b[1;32m    877\u001b[0m \u001b[38;5;66;03m# run the first trace but we should fail if variables are created.\u001b[39;00m\n\u001b[0;32m--> 878\u001b[0m results \u001b[38;5;241m=\u001b[39m tracing_compilation\u001b[38;5;241m.\u001b[39mcall_function(\n\u001b[1;32m    879\u001b[0m     args, kwds, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_variable_creation_config\n\u001b[1;32m    880\u001b[0m )\n\u001b[1;32m    881\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_created_variables:\n\u001b[1;32m    882\u001b[0m   \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCreating variables on a non-first call to a function\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    883\u001b[0m                    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m decorated with tf.function.\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py:139\u001b[0m, in \u001b[0;36mcall_function\u001b[0;34m(args, kwargs, tracing_options)\u001b[0m\n\u001b[1;32m    137\u001b[0m bound_args \u001b[38;5;241m=\u001b[39m function\u001b[38;5;241m.\u001b[39mfunction_type\u001b[38;5;241m.\u001b[39mbind(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[1;32m    138\u001b[0m flat_inputs \u001b[38;5;241m=\u001b[39m function\u001b[38;5;241m.\u001b[39mfunction_type\u001b[38;5;241m.\u001b[39munpack_inputs(bound_args)\n\u001b[0;32m--> 139\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m function\u001b[38;5;241m.\u001b[39m_call_flat(  \u001b[38;5;66;03m# pylint: disable=protected-access\u001b[39;00m\n\u001b[1;32m    140\u001b[0m     flat_inputs, captured_inputs\u001b[38;5;241m=\u001b[39mfunction\u001b[38;5;241m.\u001b[39mcaptured_inputs\n\u001b[1;32m    141\u001b[0m )\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py:1322\u001b[0m, in \u001b[0;36mConcreteFunction._call_flat\u001b[0;34m(self, tensor_inputs, captured_inputs)\u001b[0m\n\u001b[1;32m   1318\u001b[0m possible_gradient_type \u001b[38;5;241m=\u001b[39m gradients_util\u001b[38;5;241m.\u001b[39mPossibleTapeGradientTypes(args)\n\u001b[1;32m   1319\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m (possible_gradient_type \u001b[38;5;241m==\u001b[39m gradients_util\u001b[38;5;241m.\u001b[39mPOSSIBLE_GRADIENT_TYPES_NONE\n\u001b[1;32m   1320\u001b[0m     \u001b[38;5;129;01mand\u001b[39;00m executing_eagerly):\n\u001b[1;32m   1321\u001b[0m   \u001b[38;5;66;03m# No tape is watching; skip to running the function.\u001b[39;00m\n\u001b[0;32m-> 1322\u001b[0m   \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_inference_function\u001b[38;5;241m.\u001b[39mcall_preflattened(args)\n\u001b[1;32m   1323\u001b[0m forward_backward \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_select_forward_and_backward_functions(\n\u001b[1;32m   1324\u001b[0m     args,\n\u001b[1;32m   1325\u001b[0m     possible_gradient_type,\n\u001b[1;32m   1326\u001b[0m     executing_eagerly)\n\u001b[1;32m   1327\u001b[0m forward_function, args_with_tangents \u001b[38;5;241m=\u001b[39m forward_backward\u001b[38;5;241m.\u001b[39mforward()\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:216\u001b[0m, in \u001b[0;36mAtomicFunction.call_preflattened\u001b[0;34m(self, args)\u001b[0m\n\u001b[1;32m    214\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mcall_preflattened\u001b[39m(\u001b[38;5;28mself\u001b[39m, args: Sequence[core\u001b[38;5;241m.\u001b[39mTensor]) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Any:\n\u001b[1;32m    215\u001b[0m \u001b[38;5;250m  \u001b[39m\u001b[38;5;124;03m\"\"\"Calls with flattened tensor inputs and returns the structured output.\"\"\"\u001b[39;00m\n\u001b[0;32m--> 216\u001b[0m   flat_outputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcall_flat(\u001b[38;5;241m*\u001b[39margs)\n\u001b[1;32m    217\u001b[0m   \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfunction_type\u001b[38;5;241m.\u001b[39mpack_output(flat_outputs)\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:251\u001b[0m, in \u001b[0;36mAtomicFunction.call_flat\u001b[0;34m(self, *args)\u001b[0m\n\u001b[1;32m    249\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m record\u001b[38;5;241m.\u001b[39mstop_recording():\n\u001b[1;32m    250\u001b[0m   \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_bound_context\u001b[38;5;241m.\u001b[39mexecuting_eagerly():\n\u001b[0;32m--> 251\u001b[0m     outputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_bound_context\u001b[38;5;241m.\u001b[39mcall_function(\n\u001b[1;32m    252\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mname,\n\u001b[1;32m    253\u001b[0m         \u001b[38;5;28mlist\u001b[39m(args),\n\u001b[1;32m    254\u001b[0m         \u001b[38;5;28mlen\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfunction_type\u001b[38;5;241m.\u001b[39mflat_outputs),\n\u001b[1;32m    255\u001b[0m     )\n\u001b[1;32m    256\u001b[0m   \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    257\u001b[0m     outputs \u001b[38;5;241m=\u001b[39m make_call_op_in_graph(\n\u001b[1;32m    258\u001b[0m         \u001b[38;5;28mself\u001b[39m,\n\u001b[1;32m    259\u001b[0m         \u001b[38;5;28mlist\u001b[39m(args),\n\u001b[1;32m    260\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_bound_context\u001b[38;5;241m.\u001b[39mfunction_call_options\u001b[38;5;241m.\u001b[39mas_attrs(),\n\u001b[1;32m    261\u001b[0m     )\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/eager/context.py:1552\u001b[0m, in \u001b[0;36mContext.call_function\u001b[0;34m(self, name, tensor_inputs, num_outputs)\u001b[0m\n\u001b[1;32m   1550\u001b[0m cancellation_context \u001b[38;5;241m=\u001b[39m cancellation\u001b[38;5;241m.\u001b[39mcontext()\n\u001b[1;32m   1551\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m cancellation_context \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m-> 1552\u001b[0m   outputs \u001b[38;5;241m=\u001b[39m execute\u001b[38;5;241m.\u001b[39mexecute(\n\u001b[1;32m   1553\u001b[0m       name\u001b[38;5;241m.\u001b[39mdecode(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mutf-8\u001b[39m\u001b[38;5;124m\"\u001b[39m),\n\u001b[1;32m   1554\u001b[0m       num_outputs\u001b[38;5;241m=\u001b[39mnum_outputs,\n\u001b[1;32m   1555\u001b[0m       inputs\u001b[38;5;241m=\u001b[39mtensor_inputs,\n\u001b[1;32m   1556\u001b[0m       attrs\u001b[38;5;241m=\u001b[39mattrs,\n\u001b[1;32m   1557\u001b[0m       ctx\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m,\n\u001b[1;32m   1558\u001b[0m   )\n\u001b[1;32m   1559\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m   1560\u001b[0m   outputs \u001b[38;5;241m=\u001b[39m execute\u001b[38;5;241m.\u001b[39mexecute_with_cancellation(\n\u001b[1;32m   1561\u001b[0m       name\u001b[38;5;241m.\u001b[39mdecode(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mutf-8\u001b[39m\u001b[38;5;124m\"\u001b[39m),\n\u001b[1;32m   1562\u001b[0m       num_outputs\u001b[38;5;241m=\u001b[39mnum_outputs,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1566\u001b[0m       cancellation_manager\u001b[38;5;241m=\u001b[39mcancellation_context,\n\u001b[1;32m   1567\u001b[0m   )\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/tensorflow/python/eager/execute.py:53\u001b[0m, in \u001b[0;36mquick_execute\u001b[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[1;32m     51\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m     52\u001b[0m   ctx\u001b[38;5;241m.\u001b[39mensure_initialized()\n\u001b[0;32m---> 53\u001b[0m   tensors \u001b[38;5;241m=\u001b[39m pywrap_tfe\u001b[38;5;241m.\u001b[39mTFE_Py_Execute(ctx\u001b[38;5;241m.\u001b[39m_handle, device_name, op_name,\n\u001b[1;32m     54\u001b[0m                                       inputs, attrs, num_outputs)\n\u001b[1;32m     55\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m core\u001b[38;5;241m.\u001b[39m_NotOkStatusException \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[1;32m     56\u001b[0m   \u001b[38;5;28;01mif\u001b[39;00m name \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "\n",
    "# Assuming cap and draw_label are defined elsewhere\n",
    "while True:\n",
    "    ret, frame = cap.read()\n",
    "    if not ret:\n",
    "        break\n",
    "\n",
    "    img = cv2.resize(frame, (224, 224))\n",
    "    img = img / 255.0\n",
    "    y_pred = detect_face_mask(img)\n",
    "\n",
    "    if y_pred <= 0.10:\n",
    "        draw_label(frame, \"Mask\", (30, 30), (0, 255, 0))\n",
    "    else:\n",
    "        draw_label(frame, \"No Mask\", (30, 30), (0, 0, 255))\n",
    "\n",
    "    cv2.imshow(\"window\", frame)\n",
    "\n",
    "    # Process GUI events and check if the window is closed\n",
    "    key = cv2.waitKey(1)  # Short delay to process events\n",
    "    if key & 0xFF == ord('x') or cv2.getWindowProperty(\"window\", cv2.WND_PROP_VISIBLE) < 1:\n",
    "        break\n",
    "\n",
    "# Release resources and close window\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "da08cf48-baee-4eee-94a2-882f415aa024",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = cv2.imread('dataset/with_mask/with_mask_2.jpg')\n",
    "image = cv2.resize(image,(224,224))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "88952786-d0f6-42ce-8bbb-8baec05c7e64",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 91ms/step\n",
      "Mask\n"
     ]
    }
   ],
   "source": [
    "pred = detect_face_mask(image)\n",
    "if pred <= 0.50:\n",
    "    print(\"Mask\")\n",
    "else:\n",
    "    print(\"No mask\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02242ec5-83ad-4c97-a958-9468a2e27a15",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
