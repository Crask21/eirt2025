# MY Computer:
# python -m scripts.train_segmentation \
#     --train_images "G:\\datasets\\eirt_output\\batch03\\rgb" \
#     --train_masks "G:\\datasets\\eirt_output\\batch03\\mask" \
#     --val_images "G:\\datasets\\eirt_output\\batch03_val\\rgb" \
#     --val_masks "G:\\datasets\\eirt_output\\batch03_val\\mask" \
#     --class_mapping "G:\\datasets\\eirt_objects\\class_id.json" \
#     --output_dir "outputs\\pixelseg\\batch03" \
#     --architecture "mobilenet" \
#     --epochs 50


# ANDREAS' Computer:
python3 -m scripts.train_segmentation \
    --train_images "/media/ap/00802B74802B6F7A/Users/andpo/Documents/EIRT/Caspers_fredags_filer/batch04/rgb" \
    --train_masks "/media/ap/00802B74802B6F7A/Users/andpo/Documents/EIRT/Caspers_fredags_filer/batch04/mask" \
    --val_images "/media/ap/00802B74802B6F7A/Users/andpo/Documents/EIRT/Caspers_fredags_filer/batch04_val/rgb" \
    --val_masks "/media/ap/00802B74802B6F7A/Users/andpo/Documents/EIRT/Caspers_fredags_filer/batch04_val/mask" \
    --class_mapping "/media/ap/00802B74802B6F7A/Users/andpo/Documents/EIRT/Caspers_fredags_filer/class_id.json" \
    --output_dir "outputs/pixelseg/batch04_resnet" \
    --architecture "resnet" \
    --epochs 50

# python -m scripts.train_segmentation \
#     --train_images "G:/datasets/eirt_output/batch03/rgb" \
#     --train_masks "G:/datasets/eirt_output/batch03/mask" \
#     --val_images "G:/datasets/eirt_output/batch03_val/rgb" \
#     --val_masks "G:/datasets/eirt_output/batch03_val/mask" \
#     --class_mapping "G:/datasets/eirt_objects/class_id.json" \
#     --output_dir "outputs/pixelseg/batch03" \
#     --epochs 50