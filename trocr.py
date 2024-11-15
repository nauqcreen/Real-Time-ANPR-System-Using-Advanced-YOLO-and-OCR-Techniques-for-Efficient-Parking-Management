from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import cv2
import pytesseract
import numpy as np
import streamlit as st
from datetime import datetime
import pandas as pd
import os
import tempfile
from models_ import yolo_v9c, yolo_v10x

# Initialize TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

# Path to the CSV file
csv_file_path = 'parking_records.csv'

# Load or initialize the parking_df DataFrame
try:
    parking_df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    parking_df = pd.DataFrame(columns=['plate_number', 'check_in', 'check_out', 'duration_hours', 'fee_vnd'])
    parking_df.to_csv(csv_file_path, index=False)

# Image pre-processing function
def preprocess_image(img):
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    smooth = cv2.GaussianBlur(img, (1, 1), 0)
    return smooth

# Function to detect plate using YOLO model
def detect_objects(img, conf=0.8, model_type='YOLOv9C'):
    model = yolo_v9c.YOLOv9C(r"/Users/dolphin/Downloads/projet/anpr/YOLO/best_v9c.pt") if model_type == 'YOLOv9C' else yolo_v10x.YOLOv10X(r"/Users/dolphin/Downloads/projet/anpr/YOLO/best_v10x.pt")
    results, output_image = model.detect(img, conf)
    return results, output_image

# Function to perform OCR using TrOCR
def ocr_trocr(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

# Check-in function
def check_in(plate_number):
    global parking_df
    new_row = pd.DataFrame({'plate_number': [plate_number], 'check_in': [datetime.now()], 'check_out': [None], 'duration_hours': [None], 'fee_vnd': [None]})
    parking_df = pd.concat([parking_df, new_row], ignore_index=True)
    parking_df.to_csv(csv_file_path, index=False)

# Check-out function
def check_out(plate_number):
    global parking_df
    check_out_time = datetime.now()
    
    # Update check-out time
    parking_df.loc[(parking_df['plate_number'] == plate_number) & (parking_df['check_out'].isnull()), 'check_out'] = check_out_time

    # Calculate duration and fee
    check_in_time = pd.to_datetime(parking_df.loc[(parking_df['plate_number'] == plate_number) & (parking_df['check_out'] == check_out_time), 'check_in'].values[0])
    duration_hours = (check_out_time - check_in_time).total_seconds() / 3600
    fee = round(duration_hours * 15000, 2)  # Fee per hour = 15000 VND

    # Save duration and fee into the DataFrame
    parking_df.loc[(parking_df['plate_number'] == plate_number) & (parking_df['check_out'] == check_out_time), 'duration_hours'] = duration_hours
    parking_df.loc[(parking_df['plate_number'] == plate_number) & (parking_df['check_out'] == check_out_time), 'fee_vnd'] = fee

    # Save the updated DataFrame to CSV
    parking_df.to_csv(csv_file_path, index=False)

    return duration_hours, fee

def is_valid_plate(plate_text):
    # Remove "-" and "." from the plate and check if it has 8 alphanumeric characters
    clean_plate = ''.join(e for e in plate_text if e.isalnum())
    return len(clean_plate) == 8, clean_plate

# Function to handle input
def handle_input():
    # Checkbox for input selection
    input_type = st.sidebar.radio(
        "Select Input Type", 
        ('Image', 'Video', 'Webcam')
    )

    # Handle image input
    if input_type == 'Image':
        uploaded_file = st.file_uploader("Upload an image to process",
                                          type=["jpg", "jpeg", "png"],
                                          help="Supported image formats: JPG, JPEG, PNG")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            st.sidebar.image(image, caption="UPLOADED IMAGE", use_column_width=True)
            return image

    # Handle video input
    elif input_type == 'Video':
        uploaded_video = st.file_uploader("Upload a video file to process", 
                                          type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_capture = cv2.VideoCapture(tfile.name)
            st.video(tfile.name)

            frames_processed = 0
            frame_skip = 50
            stop_processing = False  # Flag to stop processing once a valid plate is detected
            detected_plate_text = None  # Store detected plate

            # Iterate through video frames
            while video_capture.isOpened():
                if stop_processing:  # Stop processing if condition is met
                    break

                success, frame = video_capture.read()
                if not success:
                    break

                if frames_processed % frame_skip == 0:
                    # Process each frame for plate detection
                    plates, output_image = detect_objects(frame, conf=0.5, model_type="YOLOv9C")

                    # Collect results for all frames and perform character recognition (OCR)
                    if plates:
                        for plate in plates:
                            x1, y1, x2, y2, conf = plate
                            # Crop the detected plate from the frame
                            cropped_plate = output_image[y1:y2, x1:x2]
                            st.image(cropped_plate, caption="Detected Plate", use_column_width=True)
                            
                            # Preprocess the cropped plate image for better OCR accuracy
                            processed_plate = preprocess_image(cropped_plate)
                            st.image(processed_plate, caption="Processed Plate for OCR", use_column_width=True)
                            
                            # Perform OCR to extract characters
                            plate_text = ocr_trocr(processed_plate)
                            st.write(f"Detected License Plate: {plate_text}")

                            # If a plate with exactly 8 alphanumeric characters is detected, save it and stop processing
                            is_valid, clean_plate = is_valid_plate(plate_text)
                            if is_valid:
                                detected_plate_text = plate_text
                                st.write(f"Stopping on detected plate: {plate_text}")
                                
                                # Check if the plate number already exists in the CSV and if it's checked out
                                existing_record = parking_df[(parking_df['plate_number'] == plate_text) & (parking_df['check_out'].isnull())]
                                if not existing_record.empty:
                                    st.sidebar.success(f"Plate {plate_text} found, recording check-out.")
                                    duration, fee = check_out(plate_text)
                                    st.sidebar.success(f"Check-out for {plate_text}\nDuration: {round(duration, 2)} hours\nFee: {fee} VND")
                                else:
                                    st.sidebar.success(f"New plate {plate_text} detected, recording check-in.")
                                    check_in(plate_text)

                                stop_processing = True
                                break

                frames_processed += 1

            # Release the video_capture resource after processing the entire video
            video_capture.release()

            # Display final detected plate (if any) and the CSV data
            if detected_plate_text:
                st.write(f"Final Detected License Plate: {detected_plate_text}")

            st.write(f"Total Frames Processed: {frames_processed}")

            # Display the contents of the CSV file
            st.subheader("Parking Records")
            csv_data = pd.read_csv(csv_file_path)
            st.dataframe(csv_data)
            
            return None
            
     # Handle webcam input
    elif input_type == 'Webcam':
        cap = cv2.VideoCapture(0)  # Open the default webcam

        if cap.isOpened():
            st.write("Webcam is active!")
            frames_processed = 0
            frame_skip = 50
            stop_processing = False
            detected_plate_text = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from webcam.")
                    break

                st.image(frame, channels="BGR", caption="Webcam Stream", use_column_width=True)

                if frames_processed % frame_skip == 0:
                    plates, output_image = detect_objects(frame, conf=0.5, model_type="YOLOv9C")
                    if plates:
                        for plate in plates:
                            x1, y1, x2, y2, conf = plate
                            cropped_plate = output_image[y1:y2, x1:x2]
                            st.image(cropped_plate, caption="Detected Plate", use_column_width=True)
                            processed_plate = preprocess_image(cropped_plate)
                            st.image(processed_plate, caption="Processed Plate for OCR", use_column_width=True)
                            plate_text = ocr_trocr(processed_plate)
                            st.write(f"Detected License Plate: {plate_text}")
                            is_valid, clean_plate = is_valid_plate(plate_text)

                            if is_valid:
                                detected_plate_text = clean_plate
                                st.write(f"Stopping on detected valid plate: {clean_plate}")

                                existing_record = parking_df[(parking_df['plate_number'] == clean_plate) & (parking_df['check_out'].isnull())]
                                if not existing_record.empty:
                                    st.sidebar.success(f"Plate {clean_plate} found, recording check-out.")
                                    duration, fee = check_out(clean_plate)
                                    st.sidebar.success(f"Check-out for {clean_plate}\nDuration: {round(duration, 2)} hours\nFee: {fee} VND")
                                else:
                                    st.sidebar.success(f"New plate {clean_plate} detected, recording check-in.")
                                    check_in(clean_plate)

                                stop_processing = True
                                break

                frames_processed += 1

                if stop_processing:
                    break

            cap.release()

            if detected_plate_text:
                st.write(f"Final Detected License Plate: {detected_plate_text}")

            st.write(f"Total Frames Processed: {frames_processed}")

            st.subheader("Parking Records")
            csv_data = pd.read_csv(csv_file_path)
            st.dataframe(csv_data)
            
            return None

    return None

# Main function
def main():
    st.set_page_config(page_title="ANPR using TrOCR", page_icon="✨", layout="centered", initial_sidebar_state="expanded")
    st.title('Automatic Number Plate Recognition With Microsoft\'s TrOCR🚘🚙')
    st.sidebar.title('Settings 😎')
    conf = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, step=0.05)
    model_type = st.sidebar.selectbox("Select Model", ("YOLOv9C", "YOLOv10X"))
    # Get input from the user
    image = handle_input()

    if image is not None:
        # Detect plates using YOLO
        plates, output_image = detect_objects(image, conf=0.5, model_type="YOLOv9C")

        if len(plates) > 0:
            for plate in plates:
                x1, y1, x2, y2, conf = plate
                # Crop the detected plate from the image
                cropped_plate = output_image[y1:y2, x1:x2]
                
                # Preprocess the cropped plate image for better OCR accuracy
                processed_img = preprocess_image(cropped_plate)
                st.sidebar.image(processed_img, caption="Cropped Plate for OCR", use_column_width=True)
                # Perform OCR to extract characters
                plate_text = ocr_trocr(processed_img)

                # Check if the plate number already exists in the CSV and if it's checked out
                existing_record = parking_df[(parking_df['plate_number'] == plate_text) & (parking_df['check_out'].isnull())]
                if not existing_record.empty:
                    st.sidebar.success(f"Plate {plate_text} found, recording check-out.")
                    duration, fee = check_out(plate_text)
                    st.sidebar.success(f"Check-out for {plate_text}\nDuration: {round(duration, 2)} hours\nFee: {fee} VND")
                else:
                    st.sidebar.success(f"New plate {plate_text} detected, recording check-in.")
                    check_in(plate_text)

                # Show plate text and confidence in sidebar
                # st.sidebar.text(f'  Plate detected: {plate_text}\n  Confidence: {conf:.2f}')
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            st.subheader("No License Plates Detected !")

        # Display output image
        st.image(output_image, caption="PROCESSED IMAGE", use_column_width=True)

        # Display the contents of the CSV file
        st.subheader("Parking Records")
        csv_data = pd.read_csv(csv_file_path)
        st.dataframe(csv_data)

if __name__ == "__main__":
    main()