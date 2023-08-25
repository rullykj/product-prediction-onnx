from sanic import Sanic
from sanic.response import text
from sanic.response import json
from sanic_ext import Extend

import numpy as np
#import cv2
from PIL import Image
import io
import base64

import json as js
from json import JSONEncoder

from typing import List

import onnxruntime as rt

app = Sanic("MyProductPredictionApp")
app.config.CORS_ORIGINS = "*"
Extend(app)

print("load model!!!")

onnx_model_path = 'model_mod.onnx'
MyOnnxModelSession = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

product_labels = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

class TopLevel:
    received: bool
    file_name: str
    file_type: str
    product: str

    def __init__(self, received: bool, file_name: str, file_type: str, product: str) -> None:
        self.received = received
        self.file_name = file_name
        self.file_type = file_type
        self.product = product
        
class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def obj_to_json_obj(obj, encoder):
    str_json = js.dumps(obj, cls=encoder)
    json_object = js.loads(str_json)
    
    return json_object

def preprocess_input_resnet50(x):
    x_temp = np.copy(x)
    
    # mean subtraction
    # already BGR in opencv
    #x_temp = x_temp[..., ::-1]
    x_temp[..., 0] -= 91
    x_temp[..., 1] -= 103
    x_temp[..., 2] -= 131
    
    return x_temp

@app.get("/")
async def hello_world(request):
    return text("Hello, world.")

@app.post("/predict")
async def predict(request):
    strBody = request.body

    jsonBody = js.loads(strBody)
    
    
    image = Image.open(io.BytesIO(base64.decodebytes(bytes(jsonBody['image'], "utf-8"))))
    img_tensor = np.asarray(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    product_img = preprocess_input_resnet50(img_tensor)
    
    # opencv
    # im_bytes = base64.b64decode(jsonBody['image'])
    # im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    # img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    # color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # color_converted = cv2.resize(color_converted, (224, 224))
    # blob = cv2.dnn.blobFromImage(color_converted)
    # product_img = np.transpose(blob, (0, 2, 3, 1))
    # product_img = preprocess_input_resnet50(product_img)
    
    preds = MyOnnxModelSession.run(['prediction'], {"input": product_img.astype(np.float32)})
        

    #print(jsonBody['operation'])
    

    #return text(jsonBody['operation'])
    
    json_object_products = obj_to_json_obj(preds, NumpyArrayEncoder)
    index = np.argmax(preds)
    print(product_labels[index])

    return json({ 
        "error": False,
        "received": True, 
        "prediction": product_labels[index],
        "product": json_object_products, 
        "product_labels": product_labels })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)