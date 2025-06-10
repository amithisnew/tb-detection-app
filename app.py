import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

def build_cbm_inference(input_shape=(224, 224, 3), num_concepts=5):
    inputs = layers.Input(shape=input_shape)
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_concepts, activation="sigmoid", dtype='float32')(x)
    return models.Model(inputs, outputs)

def build_bnn():
    inputs = layers.Input(shape=(5,), name="input")
    x = layers.Dense(128, activation="relu", name="dense_1")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return models.Model(inputs, outputs, name="bnn")

@st.cache_resource
def load_models():
    cbm_model = build_cbm_inference()
    cbm_model.load_weights("/content/drive/MyDrive/TB_Detection_Project/models/final_model.weights.h5")
    bnn_model = build_bnn()
    bnn_model.load_weights("/content/drive/MyDrive/TB_Detection_Project/models/bnn_model.weights.h5")
    return cbm_model, bnn_model

cbm_model, bnn_model = load_models()

st.title("TB Detection System")
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((224, 224))
    st.image(img, caption="Uploaded Image", width=224)
    tensor = np.array(img)[np.newaxis, ..., np.newaxis].astype(np.float32)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-7)
    tensor = np.repeat(tensor, 3, axis=-1)
    concept_preds = cbm_model.predict(tensor)
    tb_pred = bnn_model.predict(concept_preds)
    st.subheader("Concept Predictions")
    concepts = ['Effusion', 'Consolidation', 'Edema', 'Atelectasis', 'Lung Opacity']
    for i, prob in enumerate(concept_preds[0]):
        st.write(f"{concepts[i]}: {prob*100:.2f}%")
    st.subheader("TB Prediction")
    tb_label = "TB_Positive" if tb_pred[0][0] >= 0.5 else "TB_Negative"
    st.write(f"{tb_label} (Probability: {tb_pred[0][0]*100:.2f}%)")