# # Inference
# python -m scripts.inference_segmentation \
#     --checkpoint "outputs/pixelseg/batch03/best_model.pth" \
#     --config "outputs/pixelseg/batch03/config.json" \
#     --image "test.jpg" \
#     --visualize



python scripts/visualize_segmentation.py \
    --checkpoint "outputs/pixelseg/batch03/best_model.pth" \
    --config "outputs/pixelseg/batch03/config.json" \
    --val_images "G:/datasets/eirt_output/batch03_val/rgb" \
    --val_masks "G:/datasets/eirt_output/batch03_val/mask" \
    --num_samples 10 \
    --interactive