# # Inference
# python -m scripts.inference_segmentation \
#     --checkpoint "outputs/pixelseg/batch03/best_model.pth" \
#     --config "outputs/pixelseg/batch03/config.json" \
#     --image "test.jpg" \
#     --visualize



# python scripts/visualize_segmentation.py \
#     --checkpoint "outputs/pixelseg/batch04/best_model.pth" \
#     --config "outputs/pixelseg/batch04/config.json" \
#     --val_images "G:/datasets/eirt_output/batch04_val/rgb" \
#     --val_masks "G:/datasets/eirt_output/batch04_val/mask" \
#     --num_samples 10 \
#     --interactive


#Andreas computer batch 04
python scripts/visualize_segmentation.py \
    --checkpoint "outputs/pixelseg/batch04_resnet/best_model.pth" \
    --config "outputs/pixelseg/batch04_resnet/config.json" \
    --val_images "/media/ap/00802B74802B6F7A/Users/andpo/Documents/EIRT/Caspers_fredags_filer/batch_real_cropped/rgb" \
    --val_masks "/media/ap/00802B74802B6F7A/Users/andpo/Documents/EIRT/Caspers_fredags_filer/batch_real_cropped/mask" \
    --num_samples 20 \
    --interactive