import os
import shutil
from ultralytics import YOLO

def categorize_images(input_folder, output_folder=None, model_path="best.pt", conf_threshold=0.40):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
        
    if output_folder is None:
        output_folder = os.path.join(input_folder, "Categorized_Drones")
    
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return
    
    # Same logic as app.py for class naming
    if len(model.names) > 6:
        model.names[6] = "Commercial Drone"
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    count = 0
    # Scan through images in the folder
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        img_path = os.path.join(input_folder, filename)
        
        try:
            # Run inference
            results = model(img_path, conf=conf_threshold, verbose=False)
            
            detected_classes = []
            if results and len(results[0].boxes) > 0:
                # Get the detection with the highest confidence
                best_conf = 0
                best_class_name = "Drone"
                
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names.get(cls_id, "Drone")
                    
                    if isinstance(class_name, str) and class_name.strip().lower() == "drone":
                        class_name = "Commercial Drone"
                        
                    if conf > best_conf:
                        best_conf = conf
                        best_class_name = class_name
                
                if best_conf >= conf_threshold:
                    detected_classes.append(best_class_name)
                    
            # Determine category folder
            if not detected_classes:
                category = "No_Drone"
            else:
                category = detected_classes[0]
                
            category_folder = os.path.join(output_folder, category)
            os.makedirs(category_folder, exist_ok=True)
            
            # Copy image to category folder
            dest_path = os.path.join(category_folder, filename)
            shutil.copy2(img_path, dest_path)
            print(f"[{category}] -> Copied {filename}")
            count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"\nCategorization complete! Processed {count} images.")
    print(f"Results saved to: {output_folder}")

if __name__ == "__main__":
    # ==========================================
    # EDIT THIS LINE TO CHANGE THE INPUT FOLDER
    # ==========================================
    INPUT_FOLDER = r"C:\Users\swaro\Downloads\project\final_project\Total_project\images"
    
    # Optional parameters
    OUTPUT_FOLDER = None  # Will create 'Categorized_Drones' inside the input folder
    MODEL_PATH = "best.pt"
    CONFIDENCE = 0.40
    
    print(f"Starting scanning folder: {INPUT_FOLDER}")
    categorize_images(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH, CONFIDENCE)
