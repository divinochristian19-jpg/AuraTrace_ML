from ultralytics import YOLO
from PIL import Image
import io

# Load your custom YOLOv8n model
model = YOLO('app/models/best.pt')  # Replace with your trained model path

CATEGORY_MAP = {
    'backpack':       1,
    'bag':            1,
    'wallet':         2,
    'smartphone':     3,
    'phone':          3,
    'mobile phone':   3,
    'laptop':         7,
    'computer':       7,
    'shoe':           13,
    'shoes':          13,
    't-shirt':        14,
    'tshirt':         14,
    'shirt':          14,
    'tumbler':        15,
    'water bottle':   15,
    'bottle':         15,
}

def identify_item(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Run YOLOv8n inference
        results = model(image, conf=0.25)  # conf threshold at 0.25
        result = results[0]
        
        if len(result.boxes) == 0:
            return {
                'success': True,
                'detected': None,
                'confidence': 0,
                'category_id': None,
                'matched_label': None,
                'top5': [],
            }
        
        # Get all detections sorted by confidence
        detections = []
        for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = result.names[int(cls_id)].lower()
            confidence = round(float(conf) * 100, 2)
            detections.append({
                'label': label,
                'confidence': confidence
            })
        
        # Sort by confidence (descending) and get top 5
        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:5]
        
        detected_label = detections_sorted[0]['label']
        detected_confidence = detections_sorted[0]['confidence']
        
        # Map to category
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
            'top5': detections_sorted,
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}