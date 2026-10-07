import cv2
import numpy as np


def get_temperature_from_colorbar(
    img,
    top,
    bottom,
    t_top,
    t_bottom
):
    """
    Create a temperature lookup table from the visible
    vertical temperature color bar.
    """

    x1, y1 = top
    x2, y2 = bottom

    x = int(round((x1 + x2) / 2))

    y_start = min(y1, y2)
    y_end = max(y1, y2)

    if y_end - y_start < 10:
        raise ValueError(
            "The two color-bar points are too close together."
        )

    half_width = 3

    palette = []
    temperatures = []

    for y in range(y_start, y_end + 1):

        xa = max(0, x - half_width)
        xb = min(img.shape[1], x + half_width + 1)

        pixel = np.median(
            img[y, xa:xb, :],
            axis=0
        )

        fraction = (
            (y - y_start)
            / (y_end - y_start)
        )

        temperature = (
            t_top
            + fraction * (t_bottom - t_top)
        )

        palette.append(pixel)
        temperatures.append(temperature)

    palette = np.asarray(
        palette,
        dtype=np.float32
    )

    temperatures = np.asarray(
        temperatures,
        dtype=np.float32
    )

    return palette, temperatures


def estimate_temperature_map(
    img,
    palette,
    temperatures
):
    """
    Convert image colors to estimated temperatures
    using the closest color in the temperature scale.
    """

    h, w = img.shape[:2]

    pixels = img.reshape(-1, 3).astype(
        np.float32
    )

    result = np.empty(
        len(pixels),
        dtype=np.float32
    )

    chunk_size = 100000

    for start in range(
        0,
        len(pixels),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(pixels)
        )

        p = pixels[start:end]

        distances = np.sum(
            (
                p[:, None, :]
                - palette[None, :, :]
            ) ** 2,
            axis=2
        )

        nearest = np.argmin(
            distances,
            axis=1
        )

        result[start:end] = (
            temperatures[nearest]
        )

    return result.reshape(h, w)


def analyze_track(
    img,
    polygon,
    temperature_map
):
    """
    Analyze only the selected track/rail region.
    """

    mask = np.zeros(
        img.shape[:2],
        dtype=np.uint8
    )

    polygon_array = np.asarray(
        polygon,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [polygon_array],
        255
    )

    values = temperature_map[
        mask > 0
    ]

    if len(values) == 0:
        raise ValueError(
            "The selected region contains no pixels."
        )

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    mean = float(np.mean(values))
    median = float(np.median(values))

    p5 = float(
        np.percentile(values, 5)
    )

    p95 = float(
        np.percentile(values, 95)
    )

    difference = maximum - minimum

    # USER'S DECISION RULE
    if difference < 5.0:

        status = "NO FAULT"

        action = (
            "No immediate attention required."
        )

    else:

        status = "FAULT DETECTED"

        action = (
            "Attention is required within 2 days."
        )

    # Minimum location
    masked_for_min = np.where(
        mask > 0,
        temperature_map,
        np.inf
    )

    min_index = np.argmin(
        masked_for_min
    )

    min_y, min_x = np.unravel_index(
        min_index,
        temperature_map.shape
    )

    # Maximum location
    masked_for_max = np.where(
        mask > 0,
        temperature_map,
        -np.inf
    )

    max_index = np.argmax(
        masked_for_max
    )

    max_y, max_x = np.unravel_index(
        max_index,
        temperature_map.shape
    )

    return {
        "mask": mask,
        "min_temp": minimum,
        "max_temp": maximum,
        "mean_temp": mean,
        "median_temp": median,
        "p5": p5,
        "p95": p95,
        "difference": difference,
        "status": status,
        "action": action,
        "min_location": (min_x, min_y),
        "max_location": (max_x, max_y),
    }


def create_result_image(
    img,
    analysis
):
    """
    Create an annotated result image.
    """

    output = img.copy()

    mask = analysis["mask"]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Selected region boundary
    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 0),
        3
    )

    min_x, min_y = (
        analysis["min_location"]
    )

    max_x, max_y = (
        analysis["max_location"]
    )

    # Minimum point
    cv2.circle(
        output,
        (min_x, min_y),
        10,
        (255, 255, 0),
        3
    )

    cv2.putText(
        output,
        f"MIN {analysis['min_temp']:.1f} C",
        (min_x + 10, min_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    # Maximum point
    cv2.circle(
        output,
        (max_x, max_y),
        10,
        (0, 0, 255),
        3
    )

    cv2.putText(
        output,
        f"MAX {analysis['max_temp']:.1f} C",
        (max_x + 10, max_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    return output
