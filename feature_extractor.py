import cv2
import numpy as np
from PIL import Image

class SketchFeatureExtractor:
    """Extract features from rough sketches"""
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
    
    def preprocess(self, image):
        """Preprocess image for feature extraction"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize
        image = cv2.resize(image, self.target_size)
        
        return image
    
    def extract_edges(self, image):
        """Extract edge features using Canny edge detection"""
        image = self.preprocess(image)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def extract_contours(self, image):
        """Extract contour features"""
        image = self.preprocess(image)
        
        # Threshold
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, contours, -1, 255, 2)
        
        return contour_img
    
    def create_mask(self, image, threshold=200):
        """Create binary mask from sketch"""
        image = self.preprocess(image)
        
        # Invert if needed (assume white background)
        if np.mean(image) > 127:
            image = 255 - image
        
        # Create binary mask
        _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def normalize(self, image):
        """Normalize image to [-1, 1] range for GAN"""
        image = self.preprocess(image)
        normalized = (image.astype(np.float32) / 127.5) - 1.0
        return normalized
    
    def denormalize(self, image):
        """Denormalize from [-1, 1] to [0, 255]"""
        denormalized = ((image + 1.0) * 127.5).astype(np.uint8)
        return denormalized
    
    def augment_sketch(self, image):
        """Apply augmentation to sketch"""
        image = self.preprocess(image)
        
        # Random rotation
        angle = np.random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((self.target_size[0]//2, self.target_size[1]//2), angle, 1)
        rotated = cv2.warpAffine(image, M, self.target_size)
        
        # Random noise
        noise = np.random.normal(0, 5, rotated.shape).astype(np.uint8)
        noisy = cv2.add(rotated, noise)
        
        return noisy
