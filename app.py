from flask import Flask, jsonify, request, send_file,render_template
import time
import os
import joblib
from backend import InferenceEngine as ie
import cv2



app = Flask(__name__)



imagex = joblib.load("./backend/s_target.pkl")
mask_count = 0

@app.route('/')
def home():
    return render_template('index.html') 


# Function to fetch the path of the images based on ID
def get_image_paths(image_id):

    image1_path = "./static/sjl/"+imagex[int(image_id)][0]
    image2_path = "./static/sjl/"+imagex[int(image_id)][1]
    return image1_path, image2_path

# Endpoint to serve a pair of images from a single ID
@app.route('/get_images', methods=['GET'])
def get_images():
    image_id = request.args.get('image_id')
    image1_path, image2_path = get_image_paths(image_id)

    return jsonify({
        "image1": image1_path,
        "image2": image2_path
    })

# Simulate generating a mask from an image (as JPG)
def generate_mask(image_path):
    global mask_count
    mask_count+=1

    mask = ie.seg_output(image_path).reshape(512,512)
   # mask=(mask>0).astype('float32')    
    image=cv2.imread(image_path)
    image=cv2.resize(image,(512,512))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    mask=image.reshape(512,512)*mask.reshape(512,512)
    mask = mask*255
    cv2.imwrite('./static/masks/mask'+str(mask_count)+'.jpg',mask.astype("uint8"))
    return './static/masks/mask'+str(mask_count)+'.jpg'
# Endpoint to generate masks from image IDs
@app.route('/generate_masks', methods=['POST'])
def generate_masks():
    data = request.json
    image1_id = data.get('image1_id')  # Same ID used for both masks
    image2_id = data.get('image2_id')  # Same ID used for both masks

    mask1_path = generate_mask(image1_id)
    mask2_path = generate_mask(image2_id)
    return jsonify({
        "mask1": mask1_path,
        "mask2": mask2_path
    })

# Simulate image comparison (returns 1 for match, 0 for no match)
def compare_images(image1_path, image2_path):
    e=ie.recg_output(image1_path,image2_path)


    return 0 if e<=0.55 else 1

# Endpoint to compare images and retrieve actual label
@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    image_id = data.get('image_id')

    image1_path, image2_path = get_image_paths(image_id)

    result = compare_images(image1_path, image2_path)
    actual = get_actual_label(image_id)

    return jsonify({"predicted": result, "actual": actual})

# Simulate fetching the actual label for the image pair
def get_actual_label(image_id):
    return imagex[int(image_id)][2]

if __name__ == '__main__':
    app.run(debug=True)
