import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO('brain_tumor_segmentation_model.pt')

def process_masks_with_confidence(results, confidence_threshold=0.5):
    """
    Process masks, filtering out low-confidence detections
    """
    if not results or len(results) == 0:
        return None, []
    
    # Get the first result
    result = results[0]
    
    # Check if masks and boxes exist
    if result.masks is None or result.boxes is None:
        st.warning("No tumor masks detected.")
        return None, []
    
    # Filter masks and boxes based on confidence threshold
    high_conf_masks = []
    high_conf_boxes = []
    
    for mask, box in zip(result.masks, result.boxes):
        # Check confidence
        conf = box.conf[0].item()
        if conf >= confidence_threshold:
            # Convert mask to numpy
            mask_np = mask.data.cpu().numpy().squeeze()
            
            # Threshold the mask
            binary_mask = (mask_np > 0.5).astype(np.uint8)
            high_conf_masks.append(binary_mask)
            high_conf_boxes.append(box)
    
    return high_conf_masks, high_conf_boxes

def visualize_segmentation(image, results, confidence_threshold=0.5):
    """
    Create a visualization of segmentation masks
    """
    # Convert image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create a copy for overlay
    overlay = image.copy()
    
    # Process masks with confidence filtering
    masks, boxes = process_masks_with_confidence(results, confidence_threshold)
    
    if masks is not None:
        for mask, box in zip(masks, boxes):
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours and fill
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_8, offset=(0, 0))
    
    return overlay

def main():
    st.title("Brain Tumor Segmentation Web App")

    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )

    # Load the model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display original image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run inference
        try:
            # Run model prediction
            results = model(image)
            
            # Visualize segmentation
            segmentation_overlay = visualize_segmentation(image, results, confidence_threshold)
            
            # Display segmentation result
            st.image(segmentation_overlay, caption="Tumor Segmentation", use_column_width=True)
            
            # Display additional information
            if results and len(results) > 0:
                result = results[0]
                
                # Filter and display high-confidence detections
                high_conf_detections = [
                    box for box in result.boxes 
                    if box.conf[0].item() >= confidence_threshold
                ]
                
                if high_conf_detections:
                    st.write("Detected Tumors:")
                    for i, box in enumerate(high_conf_detections):
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        st.write(f"Tumor {i+1} - Confidence: {conf:.2f}, Class: {model.names[cls]}")
                else:
                    st.write("No tumors detected above the confidence threshold.")
                
                # Check for high-confidence masks
                _, high_conf_boxes = process_masks_with_confidence(results, confidence_threshold)
                if high_conf_boxes:
                    st.write(f"Number of tumor regions detected: {len(high_conf_boxes)}")
                else:
                    st.write("No tumor regions detected above the confidence threshold.")
        
        except Exception as e:
            st.error(f"Error during inference: {e}")
            # Print full error for debugging
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()