import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import urllib.request
import io

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

CATEGORY_MAP = {
    'backpack':       1,
    'bag':            1,
    'handbag':        1,
    'purse':          1,
    'wallet':         2,
    'cell phone':     3,
    'mobile phone':   3,
    'phone':          3,
    'book':           4,
    'notebook':       4,
    'umbrella':       5,
    'water bottle':   6,
    'bottle':         6,
    'laptop':         7,
    'computer':       7,
    'keyboard':       7,
    'watch':          8,
    'wristwatch':     8,
    'sunglasses':     9,
    'glasses':        9,
    'earbuds':        10,
    'headphones':     10,
    'airpods':        10,
    'id card':        11,
    'card':           11,
    'key':            12,
    'keys':           12,
}

def identify_item(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)

        categories = weights.meta['categories']

        results = []
        for prob, idx in zip(top5_prob, top5_idx):
            label = categories[idx.item()].lower()
            results.append({
                'label': label,
                'confidence': round(prob.item() * 100, 2)
            })

        detected_label = results[0]['label']
        detected_confidence = results[0]['confidence']

        category_id = None
        matched_label = None
        for key, cid in CATEGORY_MAP.items():
            if key in detected_label or detected_label in key:
                category_id = cid
                matched_label = key
                break

        return {
            'success': True,
            'detected': detected_label,
            'confidence': detected_confidence,
            'category_id': category_id,
            'matched_label': matched_label,
            'top5': results,
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}