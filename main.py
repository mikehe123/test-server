from fastapi import FastAPI, File, UploadFile, Form
import httpx
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/epic-demo/skaranamtxtfix/rykhowbuhlvp/ocr_api")
async def ocr_api(img: UploadFile = File(...), 
                  mode: int = Form(0), 
                  is_visual: bool = Form(False), 
                  thres_vis_text: int = Form(10), 
                  is_full_recog: bool = Form(False)):

    file = {'img': (img.filename, await img.read(), img.content_type)}
    data = {
        'mode': mode ,
        'is_visual': is_visual,
        'thres_vis_text': thres_vis_text,
        'is_full_recog': is_full_recog
    }

    async with httpx.AsyncClient() as client:
        response = await client.post('https://session-168539c4-7ee0-4bc3-b4da-00d4dc71097f.devbox.training.adobesensei.io/OCRv3/ocr_api', files=file, data=data)
        response_data = response.json()

    def get_box_details(coords: List[int]):
        print(coords)
        # Ensure that the coordinates are integers
        coords = [int(coord) for coord in coords]

        # Convert the coordinates into a numpy array and reshape
        points = np.array(coords, dtype=np.float32).reshape((-1, 2))

        # Calculate the minimum area rectangle
        rect = cv2.minAreaRect(points)
        (x, y), (width, height), angle = rect

        # Width and height should be positive
        width, height = abs(width), abs(height)

        # Adjust angle for a more intuitive representation
        if width < height:
            angle += 90
            width, height = height, width

        return x, y, width, height, angle


    # Extracting boxes
    boxes = response_data.get("outputs", {}).get("box", [])
    boxes_details = [get_box_details(box) for box in boxes]

    # Sorting boxes by area (width * height) in descending order
    sorted_boxes_details = sorted(boxes_details, key=lambda x: x[2] * x[3], reverse=True)
    # for each box, make the value fixed to decimal point 2
    sorted_boxes_details = [[round(x, 2) for x in box] for box in sorted_boxes_details] 
    # for each box, if angle is negative, add 180 to it
    sorted_boxes_details = [[box[0], box[1], box[2], box[3], box[4] + 180 if box[4] < 0 else box[4]] for box in sorted_boxes_details]

    for box in sorted_boxes_details:
        if box[4] == -0:
            box[4] = 0
        elif box[4] < 0:
            box[4] += 180
        elif box[4] > 90:
            box[4] = box[4] - 180
     

    print(sorted_boxes_details)
    return sorted_boxes_details