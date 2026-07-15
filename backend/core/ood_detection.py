import numpy as np

def is_valid_brain_scan(image_array: np.ndarray) -> bool:
    """
    Validates if an uploaded image is likely a medical brain scan.
    Expects a numpy array of shape (H, W, 3) with values 0-255.
    """
    # Check 1: Grayscale Variance
    # Medical scans are grayscale. Even if saved as RGB, R, G, and B are nearly identical.
    r_mean = np.mean(image_array[:, :, 0])
    g_mean = np.mean(image_array[:, :, 1])
    b_mean = np.mean(image_array[:, :, 2])
    
    # Calculate how different the colors are. 
    # Normal photos have high variance. Scans have a variance close to 0.
    channel_variance = np.var([r_mean, g_mean, b_mean])
    
    if channel_variance > 50:
        return False  # It's a colorful image (e.g., a car, a cat, a landscape)

    # Check 2: Dark Background Ratio
    # CT/MRI scans have a large black background around the brain tissue.
    # We check if at least 15% of the image is very dark (pixel value < 20)
    total_pixels = image_array.shape[0] * image_array.shape[1] * image_array.shape[2]
    dark_pixels = np.sum(image_array < 20)
    dark_ratio = dark_pixels / total_pixels
    
    if dark_ratio < 0.15:
        return False  # Less than 15% dark space means it's likely a normal photo

    # If it passes both tests, it's highly likely to be a brain scan
    return True