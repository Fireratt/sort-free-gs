# cameras.bin
uint64 num_cameras
for each camera:
    int32 camera_id
    int32 model_id
    uint64 width
    uint64 height
    double params[num_params]   ← 关键
# Images.bin
uint64 num_images
for each image:
    int32 image_id
    4x double qvec
    3x double tvec
    int32 camera_id
    string image_name (null-terminated)
    uint64 num_points2D
    repeat:
        double x, y, int64 point3D_id