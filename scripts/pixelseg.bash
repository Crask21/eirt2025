python -m scripts.train_segmentation \
    --train_images "G:\\datasets\\eirt_output\\batch03\\rgb" \
    --train_masks "G:\\datasets\\eirt_output\\batch03\\mask" \
    --val_images "G:\\datasets\\eirt_output\\batch03_val\\rgb" \
    --val_masks "G:\\datasets\\eirt_output\\batch03_val\\mask" \
    --class_mapping "G:\\datasets\\eirt_objects\\class_id.json" \
    --output_dir "outputs\\pixelseg\\batch03" \
    --architecture "mobilenet" \
    --epochs 50

# python -m scripts.train_segmentation \
#     --train_images "G:/datasets/eirt_output/batch03/rgb" \
#     --train_masks "G:/datasets/eirt_output/batch03/mask" \
#     --val_images "G:/datasets/eirt_output/batch03_val/rgb" \
#     --val_masks "G:/datasets/eirt_output/batch03_val/mask" \
#     --class_mapping "G:/datasets/eirt_objects/class_id.json" \
#     --output_dir "outputs/pixelseg/batch03" \
#     --epochs 50