import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    min_scale_temp: float = 7.0,
    max_scale_temp: float = 40.0,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
    threshold: float = 5.0
) -> dict:
    """
    Analyzes an RGB thermal image to calculate temperatures and fault status.

    Parameters
    ----------
    image : numpy.ndarray
        RGB thermal image array.
    min_scale_temp : float
        Minimum temperature corresponding to calibration scale in °C.
    max_scale_temp : float
        Maximum temperature corresponding to calibration scale in °C.
    low_percentile : float
        Lower percentile threshold to exclude noise/extreme pixels.
    high_percentile : float
        Upper percentile threshold to exclude noise/extreme pixels.
    threshold : float
        Temperature difference threshold (°C) that triggers a fault.

    Returns
    -------
    dict
        Dictionary containing extracted temperature metrics and fault evaluation.
    """
    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32)

    # Calculate percentile pixel bounds to handle noise and artifacts
    low_pixel = np.percentile(gray_float, low_percentile)
    high_pixel = np.percentile(gray_float, high_percentile)

    # Check for uniform or low-variation image
    if high_pixel <= low_pixel:
        raise ValueError(
            "The image does not contain enough temperature variation for analysis."
        )

    # Map normalized pixel values to temperature scale
    normalized = (gray_float - low_pixel) / (high_pixel - low_pixel)
    temperature = min_scale_temp + normalized * (max_scale_temp - min_scale_temp)

    # Clip values within scale limits
    temperature = np.clip(temperature, min_scale_temp, max_scale_temp)

    # Calculate temperature stats
    min_temperature = float(np.min(temperature))
    max_temperature = float(np.max(temperature))
    temperature_difference = max_temperature - min_temperature
    mean_temperature = float(np.mean(temperature))

    # Fault decision logic
    if temperature_difference < threshold:
        status = "NO FAULT"
        action = "No immediate action required."
    else:
        status = "FAULT"
        action = "Attention required within 2 days."

    return {
        "min_temperature": min_temperature,
        "max_temperature": max_temperature,
        "temperature_difference": temperature_difference,
        "mean_temperature": mean_temperature,
        "threshold": threshold,
        "status": status,
        "action": action,
        "low_percentile": low_percentile,
        "high_percentile": high_percentile
    }
