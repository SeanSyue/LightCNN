python extract_features.py \
--resume ../LightCNN_29Layers_checkpoint.pth.tar \
--model LightCNN-29 \
--img_list ../../Insightface-original/IJB_release/IJBC/meta/ijbc_name_5pts_score.txt \
--root_path ../../Insightface-original/IJB_release/IJBC/affine-112X112 \
--save_path ../save_path_ijbc_affine

# python extract_features_megaface.py \
# --resume ../LightCNN_29Layers_checkpoint.pth.tar \
# --model LightCNN-29 \
# --img_list ../../Insightface-original/megaface_testpack_v1.0/data/facescrub_lst \
# --root_path ../../Insightface-original/megaface_testpack_v1.0/data/facescrub_images \
# --save_path ../save_path_facescrub

# python extract_features_megaface.py \
# --resume ../LightCNN_29Layers_checkpoint.pth.tar \
# --model LightCNN-29 \
# --img_list ../../Insightface-original/megaface_testpack_v1.0/data/megaface_lst \
# --root_path ../../Insightface-original/megaface_testpack_v1.0/data/megaface_images \
# --save_path ../save_path_megaface