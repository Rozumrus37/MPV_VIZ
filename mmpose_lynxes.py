import argparse
import os

import numpy as np
import pandas as pd
import torch
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from tqdm.auto import tqdm


def get_args_params():
    """Load script arguments using argparse."""
    parser = argparse.ArgumentParser("Add name and keypoints")
    parser.add_argument("id", type=str, help="Directory with images")
    parser.add_argument("od", type=str, help="Output directory")
    parser.add_argument("ovd", type=str, help="Output directory for visualisation")
    parser.add_argument(
        "--bbox-parquet",
        type=str,
        required=True,
        help="Parquet file with bounding boxes",
    )
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="If set, the method will create images with detected poses",
    )
    parser.set_defaults(show_results=False)

    return parser.parse_args()


def _group_to_bbox_list(group) -> list:
    """Create a list of bounding boxes from a group in groupby call.

    Parameters
    ----------
    group
        A dataframe containing bounding boxes.

    Returns
    -------
    A list of bounding boxes in a format [{"bbox": [xmin, ymin, xmax, ymax, conf]}, ..].
    """
    group = group.copy()
    group["conf"] = 1.0
    values = group[["xmin", "ymin", "xmax", "ymax", "conf"]].values
    values = values.astype(np.float32)
    bbox_list = [{"bbox": x} for x in values]
    return bbox_list


pose_point_names = {
    0: "left-eye",
    1: "right-eye",
    2: "left-earbase",
    3: "right-earbase",
    4: "nose",
    5: "throat",
    6: "tailbase",
    7: "withers",
    8: "L-F-elbow",
    9: "R-F-elbow",
    10: "L-B-knee",
    11: "R-B-knee",
    12: "L-F-wrist",
    13: "R-F-wrist",
    14: "L-B-ankle",
    15: "R-B-ankle",
    16: "L-F-paw",
    17: "R-F-paw",
    18: "L-B-paw",
    19: "R-B-paw",
}


def main(input_dir, output_dir, output_vis_dir, bbox_parquet, show_results=False):
    """Run MMPose detection."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # build the pose model from a config file and a checkpoint file
    pose_config = "scripts/mmpose/hrnet_w48_animalpose_256x256.py"
    pose_checkpoint = (
        "https://download.openmmlab.com/mmpose/"
        + "animal/hrnet/hrnet_w48_animalpose_256x256-34644726_20210426.pth"
    )
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)

    dataset = pose_model.cfg.data["test"]["type"]

    # (optional) load table with bounding boxes from parquet file
    bbox_df = pd.read_parquet(bbox_parquet)
    image2bbox = bbox_df.groupby("image_name").apply(_group_to_bbox_list).to_dict()

    # get list of image paths to process
    image_paths = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    print(f"Running pose detection on {len(image_paths)} images.")

    lynx_poses = []
    lynx_boxes = []
    lynx_images = []
    count_bboxes_from_file_used = 0
    for image_path in tqdm(image_paths):
        if image_path.endswith(("jpg", "jpeg", "png")):
            image_name = os.path.basename(image_path)

            # get bounding box if available
            bboxes = image2bbox.get(image_name)
            pose_results = []
            if bboxes is not None:
                count_bboxes_from_file_used += 1

                # optional
                return_heatmap = False

                # e.g. use ('backbone', ) to return backbone feature
                output_layer_names = None

                bbox_thr = 0.1
                pose_results, returned_outputs = inference_top_down_pose_model(
                    pose_model,
                    image_path,
                    bboxes,
                    bbox_thr=bbox_thr,
                    format="xyxy",
                    dataset=dataset,
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names,
                )

            if len(pose_results) > 0:
                for pose_result in pose_results:
                    lynx_boxes.append(pose_result["bbox"])
                    lynx_poses.append(pd.Series(pose_result["keypoints"].tolist()))
                    lynx_images.append(image_path)
            else:
                lynx_boxes.append(None)
                lynx_poses.append(pd.Series([None for x in range(20)]))
                lynx_images.append(image_path)

            # show the results
            if show_results:
                vis_dir = (
                    f"{output_vis_dir}/{os.path.basename(os.path.dirname(image_path))}"
                )
                vis_pose_result(
                    pose_model,
                    image_path,
                    pose_results,
                    dataset=dataset,
                    kpt_score_thr=0.1,
                    radius=5,
                    thickness=2,
                    show=False,
                    out_file=f"{vis_dir}/{image_name}",
                )
    print("Pose detection finished.")
    if bbox_parquet is not None:
        print(
            "Used {} ({}) bounding boxes from '{}' file. ".format(
                count_bboxes_from_file_used,
                count_bboxes_from_file_used / len(image_paths),
                bbox_parquet,
            )
        )

    df = pd.DataFrame(lynx_poses).rename(columns=pose_point_names)
    print(df.shape)
    df["image_path"] = lynx_images
    df["bbox"] = lynx_boxes
    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(os.path.join(output_dir, "lynx_poses_mmpose.parquet"))


if __name__ == "__main__":
    args = get_args_params()
    main(args.id, args.od, args.ovd, args.bbox_parquet, args.show_results)
