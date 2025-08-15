#!/usr/bin/env python3
"""Create a test image for waifu2x optimization testing"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    # Create a test image with various elements
    width, height = 512, 512
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Add some colorful shapes and patterns
    # Gradient background
    for y in range(height):
        for x in range(width):
            r = int(255 * (x / width))
            g = int(255 * (y / height))
            b = int(255 * ((x + y) / (width + height)))
            image.putpixel((x, y), (r, g, b))
    
    # Add circles
    for i in range(5):
        x = 100 + i * 80
        y = 100 + i * 80
        draw.ellipse([x-50, y-50, x+50, y+50], fill=(255, 255-i*50, i*50))
    
    # Add rectangles
    for i in range(3):
        x1, y1 = 50 + i * 100, 300
        x2, y2 = x1 + 80, y1 + 80
        draw.rectangle([x1, y1, x2, y2], fill=(i*100, 255, 100))
    
    # Add text
    try:
        # Try to use a system font
        font = ImageFont.load_default()
        draw.text((50, 50), "Waifu2x Test Image", fill=(0, 0, 0), font=font)
        draw.text((50, 450), "GPU Optimization Test", fill=(255, 255, 255), font=font)
    except:
        # Fallback to basic text
        draw.text((50, 50), "Waifu2x Test", fill=(0, 0, 0))
    
    return image

if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(test_dir, "test.png")
    
    image = create_test_image()
    image.save(test_image_path, "PNG")
    print(f"Test image created: {test_image_path}")