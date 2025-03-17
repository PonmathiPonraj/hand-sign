import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import streamlit as st
import time
import os

# Ensure the directory exists
capture_directory = "Captured_Images"
if not os.path.exists(capture_directory):
    os.makedirs(capture_directory)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Hand Sign Classification",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Streamlit UI
st.title("Hand Sign Classification App")
st.write("This app uses a pre-trained model to classify hand signs in real-time.")

# Initialize Hand Detector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["A", "B", "C","PEACE"]  # Ensure labels match your model's output

# Sidebar Controls
st.sidebar.title("📷 Webcam Controls")
if "webcam_enabled" not in st.session_state:
    st.session_state.webcam_enabled = False

# Toggle Button for Webcam
if st.sidebar.button("🎥 Start Webcam" if not st.session_state.webcam_enabled else "🛑 Stop Webcam"):
    st.session_state.webcam_enabled = not st.session_state.webcam_enabled

# Webcam Status Display
if st.session_state.webcam_enabled:
    st.sidebar.success("Webcam is ON")
else:
    st.sidebar.error("Webcam is OFF")

# Button to capture image
if st.sidebar.button("📸 Capture & Save Image"):
    st.session_state.capture_image = True

# Add a state for capturing images
if "capture_image" not in st.session_state:
    st.session_state.capture_image = False

# Placeholder for the video feed
FRAME_WINDOW = st.image([])

# Main Loop for Webcam Feed
if st.session_state.webcam_enabled:
    cap = cv2.VideoCapture(0)

    # Reduce the camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.error("Failed to capture image from webcam.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure the bounding box is within the frame
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

            imgCrop = img[y1:y2, x1:x2]

            # Check if imgCrop is valid
            if imgCrop.size == 0:
                continue  # Skip this frame if imgCrop is empty

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Debugging: Ensure index is within the valid range
            if 0 <= index < len(labels):
                text_label = labels[index]
            else:
                text_label = "Unknown"
                print(f"Warning: Invalid index {index}, expected range 0-{len(labels)-1}")

            # Ensure text position is within the frame
            text_y = max(30, y1 - 26)  # Prevent text from going outside the image
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)
            cv2.putText(imgOutput, text_label, (x1, text_y), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        # Convert the image to RGB format for displaying in Streamlit
        imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)

        # Resize the frame to a smaller size for display
        display_width = 400  # Set the desired display width
        scale_factor = display_width / imgOutput.shape[1]
        display_height = int(imgOutput.shape[0] * scale_factor)
        imgOutput = cv2.resize(imgOutput, (display_width, display_height))

        # Save the image if the capture button is pressed
        if st.session_state.capture_image:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_filename = os.path.join(capture_directory, f"hand_sign_{timestamp}.png")
            cv2.imwrite(image_filename, cv2.cvtColor(imgOutput, cv2.COLOR_RGB2BGR))
            st.success(f"✅ Image saved successfully as {image_filename}")
            st.session_state.capture_image = False

        # Display the frame in the Streamlit app
        FRAME_WINDOW.image(imgOutput, channels="RGB", use_container_width=True)

        # Break the loop if 'q' is pressed (for local testing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    st.info("👆 Enable the webcam from the sidebar to start the feed.")
