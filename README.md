# Sign Language Recognition 🤟

A user-friendly web app for American Sign Language (ASL) recognition using a trained Keras model. Upload images or a sequence of frames to get instant predictions of ASL letters, words, or sentences.

---

## 🚀 Features
- **Single Image Prediction:** Upload a single image of a hand sign and get the predicted ASL letter.
- **Sequence Prediction:** Upload multiple images (frames) and get the predicted sequence as a word or sentence (letters shown in a single line).
- **Modern UI:** Clean, tabbed interface with sidebar instructions and styled results.
- **Easy to Use:** No coding required—just upload your images and see results instantly.

---

## 🛠️ Setup Instructions

### 1. **Clone or Download the Repository**
Place your trained Keras model file (`sign_language_model.keras`) in the project directory.

### 2. **Install Requirements**
```bash
pip install -r requirements.txt
```

### 3. **Run the App**
```bash
streamlit run app.py
```

---

## 📷 Usage

### **Single Image**
- Go to the **Single Image** tab.
- Upload a photo of a hand sign (JPG/PNG).
- The app will display the predicted letter and confidence.

### **Sequence of Frames**
- Go to the **Sequence of Frames** tab.
- Upload multiple images (frames) in order.
- The app will show thumbnails and display the predicted sequence as a word or sentence (letters in a single line).

---

## 🧠 Model Information
- The app uses a Keras model trained on 26 ASL letters (A–Z, except J & Z), plus 'del', 'nothing', and 'space'.
- Model input: 64x64 grayscale images.
- You can retrain or replace the model as needed.

---

## 🖐️ Hand Detection (Future Work)
- YOLO-based hand detection is **not yet integrated**.
- In the future, YOLO can be used to filter for hands before ASL prediction, improving robustness.

---

## 🙏 Credits
- Built with [Streamlit](https://streamlit.io/) and [TensorFlow/Keras](https://www.tensorflow.org/).
- Model and UI by [Your Name].

---

## 📬 Feedback
For questions or suggestions, please open an issue or contact the author.

## 🎥 Demo

[![Watch the demo](demo-thumbnail.png)](demo.mp4)

> Click the image above to watch the demo video. 