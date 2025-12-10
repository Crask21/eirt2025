from src.pixelsegmentation.segmentation import PixelSegmenter

img = "G:\\datasets\\eirt_output\\batch03\\rgb\\Image0001.png"

segmenter = PixelSegmenter()
mask = segmenter.segment(img)
original, mask, overlay = segmenter.segment_and_visualize(img)