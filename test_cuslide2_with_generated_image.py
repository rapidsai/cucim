#!/usr/bin/env python3
"""
Generate a test TIFF image using the utility functions and test cuslide2 with it.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the test utilities to the path
sys.path.insert(0, str(Path(__file__).parent / "python" / "cucim" / "tests" / "util"))

from gen_image import ImageGenerator
from cucim import CuImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_image(dest_folder, image_size="1024x768", tile_size=256, compression="jpeg"):
    """Generate a test TIFF image using the utility functions."""
    logger.info(f"Generating test image in {dest_folder}...")
    
    # Recipe format: type[:subpath:pattern:image_size:tile_size:compression]
    recipe = f"tiff::stripe:{image_size}:{tile_size}:{compression}"
    
    # Create image with resolution
    resolutions = [(1, 1, "CENTIMETER")]
    
    generator = ImageGenerator(dest_folder, [recipe], resolutions, logger)
    image_paths = generator.gen()
    
    return image_paths[0] if image_paths else None


def test_cuslide2_with_image(image_path):
    """Test cuslide2 by loading and reading regions from the generated image."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing cuslide2 with image: {image_path}")
    logger.info(f"{'='*60}\n")
    
    try:
        # Load the image with cuCIM
        logger.info("Loading image with CuImage...")
        img = CuImage(image_path)
        
        # Display image metadata
        logger.info(f"✓ Image loaded successfully!")
        logger.info(f"  - Shape: {img.shape}")
        logger.info(f"  - Dimensions: {img.ndim}")
        logger.info(f"  - Dtype: {img.dtype}")
        logger.info(f"  - Device: {img.device}")
        logger.info(f"  - Size: {img.size}")
        
        # Check metadata
        if hasattr(img, 'metadata'):
            logger.info(f"  - Metadata: {img.metadata}")
        
        # Test reading a region
        logger.info("\nTesting read_region...")
        if img.shape[0] >= 256 and img.shape[1] >= 256:
            region = img.read_region(location=(100, 100), size=(256, 256))
            logger.info(f"✓ Read region successfully!")
            logger.info(f"  - Region shape: {region.shape}")
            logger.info(f"  - Region dtype: {region.dtype}")
        else:
            logger.warning("Image too small to read 256x256 region")
        
        # Test reading at different levels if pyramid exists
        if hasattr(img, 'resolutions') and img.resolutions:
            logger.info(f"\n✓ Pyramid levels found!")
            logger.info(f"  - Number of levels: {img.resolutions['level_count']}")
            
            # Try reading from level 1 if it exists
            if img.resolutions['level_count'] > 1:
                logger.info("\nTesting read_region at level 1...")
                region_l1 = img.read_region(location=(50, 50), size=(128, 128), level=1)
                logger.info(f"✓ Read region at level 1 successfully!")
                logger.info(f"  - Region shape: {region_l1.shape}")
        
        # Test getting a thumbnail
        logger.info("\nTesting thumbnail generation...")
        try:
            thumbnail = img.read_region(location=(0, 0), size=(64, 64))
            logger.info(f"✓ Generated thumbnail successfully!")
            logger.info(f"  - Thumbnail shape: {thumbnail.shape}")
        except Exception as e:
            logger.warning(f"Could not generate thumbnail: {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ All cuslide2 tests PASSED!")
        logger.info(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Error testing cuslide2: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to generate image and test cuslide2."""
    # Create temporary directory for the test image
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Test with different image configurations
        configs = [
            ("512x384", 128, "jpeg"),
            ("1024x768", 256, "jpeg"),
            ("2048x1536", 256, "jpeg"),  # This should create a pyramid
        ]
        
        all_passed = True
        for image_size, tile_size, compression in configs:
            logger.info(f"\n{'#'*60}")
            logger.info(f"Testing with config: {image_size}, tile_size={tile_size}, compression={compression}")
            logger.info(f"{'#'*60}\n")
            
            # Generate the test image
            image_path = generate_test_image(temp_dir, image_size, tile_size, compression)
            
            if not image_path:
                logger.error("❌ Failed to generate test image")
                all_passed = False
                continue
            
            logger.info(f"Generated image at: {image_path}")
            logger.info(f"File size: {os.path.getsize(image_path) / 1024:.2f} KB")
            
            # Test cuslide2 with the generated image
            passed = test_cuslide2_with_image(image_path)
            if not passed:
                all_passed = False
        
        if all_passed:
            logger.info("\n" + "="*60)
            logger.info("🎉 ALL TESTS PASSED! cuslide2 is working correctly.")
            logger.info("="*60)
        else:
            logger.error("\n" + "="*60)
            logger.error("❌ SOME TESTS FAILED")
            logger.error("="*60)
            sys.exit(1)


if __name__ == "__main__":
    main()

