# thermal_analyzer.py

import cv2
import numpy as np


def analyze_thermal_image(
    image,
    min_scale_temp,
    max_scale_temp
):
    """
    Analyze the complete thermal image.

    Parameters
    ----------
    image : numpy.ndarray
        RGB thermal image.

    min_scale_temp : float
        Minimum temperature represented by the thermal color scale.

    max_scale_temp : float
        Maximum temperature represented by the thermal color scale.

    Returns
    -------
    dict
        Minimum temperature, maximum temperature,
        temperature difference, mean temperature and status.
    """

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert pixel intensity to 0-1
    normalized = gray.astype(np.float32) / 255.0

    # Convert intensity to temperature
    temperature = (
        min_scale_temp
        + normalized * (max_scale_temp - min_scale_temp)
    )

    # ------------------------------------------------
    # WHOLE IMAGE ANALYSIS
    # ------------------------------------------------

    min_temperature = float(np.min(temperature))
    max_temperature = float(np.max(temperature))

    temperature_difference = (
        max_temperature - min_temperature
    )

    mean_temperature = float(np.mean(temperature))

    # ------------------------------------------------
    # FAULT DECISION
    # ------------------------------------------------

    threshold = 5.0

    if temperature_difference < threshold:

        status = "NO FAULT"
        action = "No immediate action required."

    else:

        status = "FAULT"
        action = "Attention required within 2 days."

    # ------------------------------------------------
    # RETURN RESULTS
    # ------------------------------------------------

    return {
        "min_temperature": min_temperature,
        "max_temperature": max_temperature,
        "temperature_difference": temperature_difference,
        "mean_temperature": mean_temperature,
        "threshold": threshold,
        "status": status,
        "action": action
    }
