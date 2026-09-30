#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, resolve_resolution
from utils.general_utils import get_available_cpu_memory_gb

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            raise ValueError(f"Could not recognize scene type at {args.source_path}. Expected COLMAP data with sparse/ or Blender data with transforms_train.json.")

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if args.image_loading_mode != None and args.image_loading_mode not in {"auto", "eager", "lazy"}:
            raise ValueError(
                "image_loading_mode must be one of {'auto', 'eager', 'lazy'}"
            )

        reference_cameras = scene_info.train_cameras or scene_info.test_cameras
        if len(reference_cameras) == 0:
            raise ValueError("No cameras found in the scene.")

        reference_camera = reference_cameras[0]
        reference_resolution = resolve_resolution(
            args, reference_camera.width, reference_camera.height, resolution_scales[0]
        )
        resident_dataset_size_in_GB = (
            1.0
            * (len(scene_info.train_cameras) + len(scene_info.test_cameras))
            * reference_resolution[0]
            * reference_resolution[1]
            * 3
            * 4
            / 1e9
        )
        available_cpu_memory_in_GB = get_available_cpu_memory_gb()

        if args.data_device == "disk":
            args.lazy_image_loading = True
            resolved_image_loading_mode = "lazy"
        elif args.image_loading_mode == "auto":
            eager_loading_budget_in_GB = (
                available_cpu_memory_in_GB * args.image_loading_auto_ram_budget_ratio
            )
            args.lazy_image_loading = (
                resident_dataset_size_in_GB > eager_loading_budget_in_GB
                or (
                    args.data_device == "cuda"
                    and resident_dataset_size_in_GB
                    > args.preload_dataset_to_gpu_threshold
                )
            )
            resolved_image_loading_mode = (
                "lazy" if args.lazy_image_loading else "eager"
            )
        else:
            args.lazy_image_loading = args.image_loading_mode == "lazy"
            resolved_image_loading_mode = args.image_loading_mode

        print(
            "Image loading mode: requested={} resolved={} data_device={} estimated_eager_dataset_size={:.2f} GB available_cpu_memory={:.2f} GB".format(
                args.image_loading_mode,
                resolved_image_loading_mode,
                args.data_device,
                resident_dataset_size_in_GB,
                available_cpu_memory_in_GB,
            )
        )

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        # @Hint: When finetuning , load_ply_from_3dgs == True . Else , it should be false
        load_ply_from_3dgs = True
        if load_ply_from_3dgs:
            self.gaussians.spatial_lr_scale = self.cameras_extent
            print("[Firerat Hint] input pointcloud path is ignored.")
            print("[Firerat Hint] Load 3DGS PLY without sort-free parameter for finetuning: output/3dgs-ply/point_cloud.ply")
            self.gaussians.load_3dgs_ply("output/3dgs-ply/point_cloud.ply")
            print(f"GS number: {gaussians.get_xyz.shape[0]}")

        else:
            print(f"""Load 3DGS PLY from model path """)
            if self.loaded_iter:  # 在这里初始化参数
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
