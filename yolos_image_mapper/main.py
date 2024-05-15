from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from PIL import Image
import requests
import json

url = "https://storage.googleapis.com/adx_media/SpaceAI_Img/Example%20Img/living%20room.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

# print(model.config.id2label)

detection_results = []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    label_name = model.config.id2label.get(label.item(), "Unknown")  # 获取标签名称
    detection_results.append({
        "score": score.item(),
        "label": label.item(),
        "label_name": label_name,
        "box": box
    })

# 按置信度降序排序
detection_results = sorted(detection_results, key=lambda x: x["score"], reverse=True)

# 打印检测结果
# for item in detection_results:
#     print(f"Detected {item['label_name']} with confidence {round(item['score'], 3)} at location {item['box']}")

# print("-------------------------")

image_mapper_data = []

for item in detection_results:
    image_mapper_data.append({
        "id": str(hash(f"{item['label_name']}-{item['box']}")),  # 唯一标识符
        "title": item['label_name'],  # 使用检测到的标签名称
        "shape": "rect",  # 假设所有区域都是矩形
        "name": str(item['label']),  # 使用标签ID作为名称
        "fillColor": "#0000ff4d",  # 半透明蓝色
        "strokeColor": "black",
        "coords": [
            item['box'][0], item['box'][1], item['box'][2], item['box'][3]
        ]
    })

# 打印或处理 image_mapper_data
# print(image_mapper_data)

file_path = 'image_mapper_data.json'
with open(file_path, 'w') as file:
    json.dump(image_mapper_data, file, indent=4)

# ------------------------------------------------------------------------------
