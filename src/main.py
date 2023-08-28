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
chair_model_path = 'resnet_chair.onnx'
sofa_model_path = 'resnet_sofa.onnx'

MyOnnxModelSession = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
chairModelSession = rt.InferenceSession(chair_model_path, providers=['CPUExecutionProvider'])
sofaModelSession = rt.InferenceSession(sofa_model_path, providers=['CPUExecutionProvider'])

product_labels = ['bed', 'chair', 'sofa', 'swivelchair', 'table']
brand_labels = ['IKEA', 'INFORMA']

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
    
    preds = MyOnnxModelSession.run(['prediction'], {"input": product_img.astype(np.float32)})
    
    json_object_products = obj_to_json_obj(preds, NumpyArrayEncoder)
    index = np.argmax(preds)
    
    if index == 0:
        brandPreds = chairModelSession.run(['prediction'], {"input": product_img.astype(np.float32)})
    elif index == 1:
        brandPreds = chairModelSession.run(['prediction'], {"input": product_img.astype(np.float32)})
    elif index == 2:
        brandPreds = sofaModelSession.run(['prediction'], {"input": product_img.astype(np.float32)})
    elif index == 3:
        brandPreds = chairModelSession.run(['prediction'], {"input": product_img.astype(np.float32)})
    elif index == 4:
        brandPreds = chairModelSession.run(['prediction'], {"input": product_img.astype(np.float32)})
    # else: 
    #     ""
    json_object_brands = obj_to_json_obj(brandPreds, NumpyArrayEncoder)
    brandIndex = np.argmax(brandPreds)     
    
    print(product_labels[index])
    print(brand_labels[brandIndex])

    return json({ 
        "error": False,
        "received": True, 
        "prediction": product_labels[index],
        "product": json_object_products, 
        "product_labels": product_labels,
        "brand_prediction": brand_labels[brandIndex],
        "brand": json_object_brands, 
        "brand_labels": brand_labels
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)