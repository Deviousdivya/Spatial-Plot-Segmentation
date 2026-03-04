import cv2
import numpy as np
import json
import os

def process_map_assignment(image_name):
    # 1. Setup Paths
    input_path = os.path.join("assets", image_name)
    output_dir = "output"
    
    # Ensure output folder exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load Image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not find '{image_name}' in assets folder.")
        return

    h, w = img.shape[:2]
    
    # 3. Image Pre-processing for Plot Detection
    # Convert to grayscale to focus on brightness [cite: 5, 22]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding: Isolate white plots (brightness > 240) [cite: 5, 22]
    # This ignores the darker roads and text [cite: 7, 8, 111]
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 4. Find Contours
    # RETR_EXTERNAL ensures we only get the outer plot boundaries [cite: 14, 36]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plot_data_list = []
    visualized_img = img.copy()

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter: Skip tiny noise or huge background areas [cite: 110]
        if area < 300: 
            continue

        # 5. Polygon Approximation (Handles 3-10+ sides and rotation) [cite: 10, 28, 112]
        # A smaller epsilon (0.005) ensures higher polygon accuracy [cite: 119]
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 6. Visualize: Draw Green Borders [cite: 14, 36]
        cv2.drawContours(visualized_img, [approx], -1, (0, 255, 0), 2)

        # 7. Extract JSON Data [cite: 38-77]
        # Pixel coordinates
        polygon_px = [pt[0].tolist() for pt in approx]
        
        # Normalized coordinates (range 0.0 to 1.0) [cite: 59-71]
        polygon_norm = [[round(p[0]/w, 6), round(p[1]/h, 6)] for p in polygon_px]
        
        # Bounding Box and Centroid [cite: 72, 76]
        x, y, bw, bh = cv2.boundingRect(cnt)
        centroid = [int(x + bw/2), int(y + bh/2)]

        plot_data_list.append({
            "id_auto": i + 1,
            "polygon_px": polygon_px,
            "polygon_norm": polygon_norm,
            "centroid_px": centroid,
            "contour_area_px": area,
            "bbox_px": [x, y, bw, bh],
            "plot_number_info": None
        })

    # 8. Save Visualized Image Output [cite: 12, 31]
    output_image_name = f"visualized_{image_name}"
    cv2.imwrite(os.path.join(output_dir, output_image_name), visualized_img)

    # 9. Save JSON Output [cite: 15, 38]
    json_output = {
        "image_path": input_path,
        "image_size": [w, h],
        "plots": plot_data_list
    }
    
    json_filename = f"{os.path.splitext(image_name)[0]}.json"
    with open(os.path.join(output_dir, json_filename), "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"Success: Processed {image_name}. Found {len(plot_data_list)} plots.")

# Execute for both images required by the assignment [cite: 131]
if __name__ == "__main__":
    # Ensure these files are in your 'assets' folder
    process_map_assignment("plot_bg_image.jpg")
    process_map_assignment("layout_plan.png") # Adjust filename to match your file