# 🧠📷 Image Caption Generator Using Deep Learning (CNN-LSTM) + Streamlit App

This is an **Image Captioning Web App** that leverages a Deep Learning-based approach to generate human-like descriptions from input images. The model utilizes a **CNN-LSTM** architecture and is trained on the **Flickr8k** dataset. The web app is developed using **Streamlit** to provide an interactive demo and deployment. I created this project as part of my undergraduate thesis.



## 📌 Project Highlights

- ✅ Extract image features using pretrained **CNN** based models (DenseNet201)
- ✅ Generate captions using **LSTM-based decoder**.
- ✅ Trained on the **Flickr8k** dataset.
- ✅ Web interface with **Streamlit**.
- ✅ Real-time captioning for uploaded images.
- ✅ Evaluated using **BLEU Score**.


---

## 🚀 How It Works

1. **Image Preprocessing**
   - Resize, normalize, and feed into a pretrained CNN (DenseNet201).
   - Save the encoded feature vector for each image.

2. **Text Preprocessing**
   - Clean captions, remove punctuations, lowercase.
   - Tokenize text and pad sequences.

3. **Model Architecture**
   - CNN Encoder (e.g., DenseNet201 without classification layer).
   - LSTM Decoder that predicts the next word based on image features and partial caption.

4. **Training**
   - Input: Image features + start of the caption.
   - Output: Next word in the caption sequence.

5. **Caption Generation**
   - Use greedy or beam search to generate complete captions from image features.

6. **Web App (Streamlit)**
   - Upload image → Process → Generate caption → Display output.


---

## 📓 How to Use the Notebook

1. Clone the repository:
```bash
git clone https://github.com/al-ghalib/Image-Captioning-CNN-LSTM-Flickr8k.git
cd Image-Captioning-CNN-LSTM-Flickr8k
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook Image_Caption_Generator.ipynb
```

4. Run each cell sequentially to:
   - Preprocess images and captions
   - Extract image features
   - Train the CNN-LSTM model
   - Generate and evaluate captions

---

## 🚀 Run Streamlit Web App

After training and saving the model:

```bash
streamlit run app.py
```

Upload an image in the browser to receive a generated caption.

---

## 📦 Dataset

**Flickr8k Dataset**
- 8,000 images, each with 5 human-annotated captions.
- Download: [Kaggle – Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Place images under `Flickr8k_Dataset/Flicker8k_Dataset/` and captions under `Flickr8k_Dataset/Flickr8k_text/`.

---

## 📈 Evaluation

- Evaluated using **BLEU Scores (1-gram to 4-gram)**.
- Assesses fluency and relevance of generated captions.

---


## ✅ Future Improvements

- Incorporate **attention mechanism**.
- Upgrade to **transformer-based architectures** (ViT, CLIP, BLIP).
- Add support for **multilingual captioning** (e.g. Bangla).
- Train on **Flickr30k** or **MS-COCO** datasets.
- Host app on **Streamlit Cloud** or **Hugging Face Spaces**.

---

## 🤝 Contributions

PRs and feedback are welcome! Feel free to fork and improve ✨

---

## 📜 License

This project is under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Flickr8k dataset by UIUC
- **TensorFlow**, **Keras**, and **Streamlit** communities for powerful open-source tools.
