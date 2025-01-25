
from ultralytics import YOLO

class YOLOv10X:
    def __init__(self, model_path):
        # Load the YOLOv10x model
        self.model = YOLO(model_path)

    def detect(self, image, conf_threshold=0.5):
        # Perform inference
        results = self.model(image)  # Inference
        
        # Extract bounding boxes (this can be either 4 or 5 values depending on the model output)
        detections = results[0].boxes.xyxy  # or results[0].boxes.xywh
        
        # Filter results based on confidence threshold
        filtered_detections = []
        for detection in detections:
            if len(detection) >= 4:  # Ensure there are at least 4 values
                x1, y1, x2, y2 = map(int, detection[:4])  # Tighter bounding boxes
                conf = float(detection[4]) if len(detection) == 5 else 1.0  # Default to 1.0 if no confidence score
                if conf >= conf_threshold:
                    # Ensure tighter bounding box around the plate (optional padding can be added if necessary)
                    filtered_detections.append((x1, y1, x2, y2, conf))
        
        return filtered_detections, results[0].plot()  # Return both detections and the annotated image
