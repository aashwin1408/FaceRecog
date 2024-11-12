# Image Segmentation and Verification for Single Face Images - Face-Streamer

## 📋 Description
**Face-Streamer** is a deep learning project that provides a real-time web application for:
- **Segmentation**: Utilizing Haar features, Histogram of Oriented Gradients (HOG), and the famous UNet architecture for image segmentation.
- **Verification**: Leveraging Siamese Networks with Contrastive Loss for face verification.

The project demonstrates these processes visually, enabling real-time segmentation and verification of single face images.

---

## ⚙️ Installation and Usage

### Prerequisites
Make sure you have **Python** installed on your system. The project runs inside a virtual environment using Python's `venv`.

### Steps to Set Up and Run:

```bash
# Step 1: Clone the repository and navigate to the project directory
git clone https://github.com/your-repository/face-streamer.git
cd face-streamer

# Step 2: virtual environment -  activate it

source bin/activate  # On Windows, use `venv\Scripts\activate`

# Step 4: Run the Flask application
python app.py

# Step 5: Access the web application in your browser
# Go to the URL below:
http://127.0.0.1:5000

# Step 6: Initialize the process
# Once the web app is running, click the "Init Button" at the bottom left of the web page to start the segmentation and verification process.
```
## 🔍 Features
- Real-time Image Segmentation using:
- Haar Features
- Histogram of Oriented Gradients (HOG)
- UNet Architecture
- Face Verification using:
   - Siamese Networks
   - Contrastive Loss

## 🛠️  Technologies Used
- Python
- Flask - for creating the web application
- #### Deep Learning Libraries for segmentation and verification
  - Tensorflow with cuda
  - opencv

