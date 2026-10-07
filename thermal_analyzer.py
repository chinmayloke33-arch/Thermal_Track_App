import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Isolates target equipment from background noise using automated masking
    and extracts actual hotspot vs ambient equipment temperatures.
    """
    # 1. Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Crop inner 80% region to strip off camera UI margins and colorbar legends
    h, w = gray.shape
    roi = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    # 3. Create a mask isolating the object from cold background air/shadows
    # Uses Otsu thresholding to separate foreground equipment from cold background
    _, object_mask = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Extract pixel values belonging strictly to the equipment
    equipment_pixels = roi[object_mask > 0]

    if len(equipment_pixels) == 0:
        equipment_pixels = roi.flatten()

    # 4. Measure ambient equipment baseline (10th percentile) vs hotspot (99th percentile)
    pixel_ambient = np.percentile(equipment_pixels, 10)
    pixel_hotspot = np.percentile(equipment_pixels, 99)
    pixel_mean = np.mean(equipment_pixels)

    # Convert intensity values to Celsius based on scale span
    scale_range = known_hot_temp - known_cold_temp
    temp_per_pixel = scale_range / 255.0

    min_temp = float(known_cold_temp + (pixel_ambient * temp_per_pixel))
    max_temp = float(known_cold_temp + (pixel_hotspot * temp_per_pixel))
    mean_temp = float(known_cold_temp + (pixel_mean * temp_per_pixel))

    # Calculate target temperature difference
    temp_diff = float(max_temp - min_temp)

    # 5. Fault decision logic
    if temp_diff < threshold:
        status = "NO FAULT"
        action = "No immediate action required."
    else:
        status = "FAULT"
        action = "Attention required within 2 days."

    return {
        "min_temperature": round(min_temp, 2),
        "max_temperature": round(max_temp, 2),
        "temperature_difference": round(temp_diff, 2),
        "mean_temperature": round(mean_temp, 2),
        "threshold": threshold,
        "status": status,
        "action": action
    }
