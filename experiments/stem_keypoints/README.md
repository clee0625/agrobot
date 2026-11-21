# Stem junction keypoints (standalone test)

Finds the junction where a tomato cluster stem leaves the main stem, and reports a cut point and stem angles.
Standalone experiment. Not part of the `agrobot_perception` build.

- `stem_inference.py`: `StemDetector.detect(bgr)`, no ROS. Keypoint R-CNN (torchvision), 4 keypoints: Mtop, JP, Mbut, Pu.
- `stem_keypoint_node.py`: subscribes to `/camera/camera/color/image_raw`, publishes JSON on `/agrobot/stem_junctions`.
- `image_folder_publisher.py`: test tool that publishes saved images in place of a camera.

## Run
    python stem_keypoint_node.py                # needs the weights file in this folder
    python image_folder_publisher.py images     # folder of .jpg files

Tested on Ubuntu 24.04, ROS 2 Jazzy, PyTorch 2.11 with a GPU, using Fast DDS.
Weights (`keypointrcnn_peduncle_aug_best.pth`, about 230 MB) are not in the repo.

## Status
2D only. Angles are in the image (0 = right, 90 = up). Cut point is a placeholder, halfway from JP to Pu.
Trained on the Wageningen peduncle dataset (CC BY): https://doi.org/10.4121/bca55ad7-da67-408d-a083-01376124f514
