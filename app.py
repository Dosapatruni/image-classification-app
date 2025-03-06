{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "793411e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Defaulting to user installation because normal site-packages is not writeable\n",
      "Requirement already satisfied: tensorflow in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (2.18.0)\n",
      "Requirement already satisfied: pandas in c:\\programdata\\anaconda3\\lib\\site-packages (2.2.2)\n",
      "Requirement already satisfied: numpy in c:\\programdata\\anaconda3\\lib\\site-packages (1.26.4)\n",
      "Requirement already satisfied: matplotlib in c:\\programdata\\anaconda3\\lib\\site-packages (3.9.2)\n",
      "Requirement already satisfied: streamlit in c:\\programdata\\anaconda3\\lib\\site-packages (1.37.1)\n",
      "Requirement already satisfied: tensorflow-intel==2.18.0 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow) (2.18.0)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.1.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=24.3.25 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (25.2.10)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.6.0)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (18.1.1)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.4.0)\n",
      "Requirement already satisfied: packaging in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (24.1)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (4.25.3)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.32.3)\n",
      "Requirement already satisfied: setuptools in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (75.1.0)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.5.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (4.11.0)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.14.1)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.70.0)\n",
      "Requirement already satisfied: tensorboard<2.19,>=2.18 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.18.0)\n",
      "Requirement already satisfied: keras>=3.5.0 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.9.0)\n",
      "Requirement already satisfied: h5py>=3.11.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.11.0)\n",
      "Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.4.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (4.51.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (1.4.4)\n",
      "Requirement already satisfied: pillow>=8 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (10.4.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from matplotlib) (3.1.2)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (16.1.0)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (13.7.1)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (8.2.3)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (3.1.43)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (6.4.1)\n",
      "Requirement already satisfied: watchdog<5,>=2.1.5 in c:\\programdata\\anaconda3\\lib\\site-packages (from streamlit) (4.0.1)\n",
      "Requirement already satisfied: jinja2 in c:\\programdata\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
      "Requirement already satisfied: toolz in c:\\programdata\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: colorama in c:\\programdata\\anaconda3\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\programdata\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow) (2025.1.31)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.18.0->tensorflow) (0.44.0)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\programdata\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\programdata\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: namex in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (0.0.8)\n",
      "Requirement already satisfied: optree in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (0.14.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\sanja\\appdata\\roaming\\python\\python312\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (0.7.2)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (3.0.3)\n"
     ]
    }
   ],
   "source": [
    "!pip install tensorflow pandas numpy matplotlib streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1e1145b3-264d-44fe-a416-de3989558683",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append(r\"C:\\Users\\sanja\\AppData\\Roaming\\Python\\Python312\\site-packages\")\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9d83e1d3-ac65-4640-a0e8-417cfe4bacd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras.models import load_model\n",
    "import streamlit as st\n",
    "import numpy as np\n",
    "from PIL import Image  # For handling images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "673f7dc7-90f8-4f7c-812a-ba49cb71a9ea",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-06 14:24:54.606 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\ProgramData\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# Streamlit Header\n",
    "st.header('Image Classification Model')\n",
    "\n",
    "# Load the trained model\n",
    "model = load_model(\"D:\\\\image classification\\\\Image_classify (1).keras\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "64ce36fc-1ffe-4f58-8b90-6021c871bdee",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Categories for classification\n",
    "data_cat = [\n",
    "    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',\n",
    "    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',\n",
    "    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',\n",
    "    'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',\n",
    "    'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',\n",
    "    'turnip', 'watermelon'\n",
    "]\n",
    "\n",
    "img_height = 180\n",
    "img_width = 180\n",
    "image_size = (img_width, img_height)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ee9ca9da-eac5-485c-a12d-40e30efa4c8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Upload image through Streamlit\n",
    "uploaded_file = st.file_uploader(\"Upload an image\", type=[\"jpg\", \"png\", \"jpeg\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    image = Image.open(uploaded_file)\n",
    "    st.image(image, caption=\"Uploaded Image\", width=200)\n",
    "    \n",
    "    # Preprocess the image\n",
    "    image = image.resize(image_size)\n",
    "    img_arr = tf.keras.preprocessing.image.img_to_array(image)\n",
    "    img_bat = tf.expand_dims(img_arr, 0)\n",
    "\n",
    "    # Make prediction\n",
    "    prediction = model.predict(img_bat)\n",
    "    score = tf.nn.softmax(prediction[0])\n",
    "\n",
    "    # Display the result\n",
    "    st.write(f'**Predicted: {data_cat[np.argmax(score)]}**')\n",
    "    st.write(f'**Confidence: {np.max(score) * 100:.2f}%**')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
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
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
