
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI 垃圾分類小幫手", page_icon="♻️")

st.title("我是你的垃圾分類小幫手 ♻️")
st.caption("請上傳一張垃圾圖片，我幫你判斷分類！")

uploaded_file = st.file_uploader("📷 上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="你上傳的圖片", use_column_width=True)

    model = tf.keras.models.load_model("model.h5")
    class_names = [
        "寶特瓶", "衛生紙", "玻璃", "紙類", "便當盒",
        "手搖飲料杯", "廢電器", "鋁罐", "藥"
    ]

    class_info = {
        "寶特瓶": "我是寶特瓶 ♻️ 請壓扁再回收！",
        "鋁罐": "我是鋁罐～請我進回收桶 ♻️",
        "玻璃": "我是玻璃罐，請丟玻璃專用回收箱 🧪",
        "紙類": "我是乾淨紙類 🧻 請回收我～",
        "衛生紙": "我是用過的衛生紙，要丟到一般垃圾桶唷 🗑️",
        "便當盒": "我是便當盒 🍱 看能否洗乾淨回收",
        "手搖飲料杯": "我是手搖杯 🧋 請分開杯蓋封膜再回收",
        "免洗餐具": "我是免洗餐具 🍴 看材質決定是否回收",
        "藥": "我是藥品 💊 請送去藥物回收點",
        "廢電器": "我是廢電器 🔌 請送去資源回收站"
    }

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    label = class_names[idx]

    st.subheader(f"🧠 我覺得這是：{label}")
    st.success(class_info.get(label, "（分類說明待補）"))
