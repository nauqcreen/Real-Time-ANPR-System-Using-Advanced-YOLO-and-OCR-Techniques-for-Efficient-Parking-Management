from ultralytics import YOLO

class YOLOv9C:
    def __init__(self, model_path):
        # Load the YOLOv9c model
        self.model = YOLO(model_path)

    def detect(self, image, conf_threshold=0.25):
        # Perform inference
        results = self.model(image)  # Inference
        
        # Extract bounding boxes (this can be either 4 or 5 values depending on the model output)
        detections = results[0].boxes.xyxy  # or results[0].boxes.xywh
        
        # Filter results based on confidence threshold
        filtered_detections = []
        for detection in detections:
            if len(detection) >= 4:  # Ensure there are at least 4 values
                x1, y1, x2, y2 = map(int, detection[:4])
                conf = float(detection[4]) if len(detection) == 5 else 1.0  # Default to 1.0 if no confidence score
                if conf >= conf_threshold:
                    filtered_detections.append((x1, y1, x2, y2, conf))
        
        return filtered_detections, results[0].plot()  # Return detections and annotated image
