import cv2
import os
import pickle
from typing import List
import matplotlib.pyplot as plt
from boxinfo import BoxInfo  # Custom class that parses a line of tracking annotation


# Root directory of the dataset (update path as needed)
#dataset_root = "E:\\dataset\\pro\\HDTM_group_activity_rec\\data\\volleyball"
dataset_root = '/kaggle/input/volleyball'


def load_tracking_annot(path):
    """
    Load tracking annotations for a single clip.

    Args:
        path (str): Path to a tracking annotation file (e.g., "clip_XXXX.txt").

    Returns:
        dict[int, list[BoxInfo]]: 
            A dictionary mapping each frame ID → list of BoxInfo objects
            representing all player bounding boxes in that frame.
    """
    with open(path, 'r') as file:
        # Initialize a dictionary: each key is a player ID (0–11), each value a list of BoxInfo objects.
        player_boxes = {idx: [] for idx in range(12)}
        frame_boxes_dct = {}

        # Parse each line in the annotation file
        for idx, line in enumerate(file):
            box_info = BoxInfo(line)  # Convert the text line into a BoxInfo object

            # Skip players with IDs > 11 (since Volleyball dataset uses up to 12 players)
            if box_info.player_ID > 11:
                continue

            # Store the parsed box info per player
            player_boxes[box_info.player_ID].append(box_info)

        # Build a frame-to-box mapping dictionary
        for player_ID, boxes_info in player_boxes.items():
            # Keep only a subset of frames (empirically chosen)
            # In this version, only frames [9, 10, 11] are kept for efficiency
            boxes_info = boxes_info[9:12]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []

                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct

# def vis_clip(annot_path, video_dir):
#     frame_boxes_dct = load_tracking_annot(annot_path)
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     for frame_id, boxes_info in frame_boxes_dct.items():
#         img_path = os.path.join(video_dir, f'{frame_id}.jpg')
#         image = cv2.imread(img_path)

#         for box_info in boxes_info:
#             x1, y1, x2, y2 = box_info.box

#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, box_info.category, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

#         cv2.imshow('Image', image)
#         cv2.waitKey(180)
#     cv2.destroyAllWindows()
def vis_clip(annot_path, video_dir):
    """
    Visualize bounding boxes for a single video clip.

    Args:
        annot_path (str): Path to the clip’s tracking annotation file.
        video_dir (str): Path to the directory containing the clip’s image frames.
    """
    frame_boxes_dct = load_tracking_annot(annot_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Loop through all frames that have annotations
    for frame_id, boxes_info in frame_boxes_dct.items():
        img_path = os.path.join(video_dir, f'{frame_id}.jpg')
        image = cv2.imread(img_path)

        # Draw each bounding box with its category label
        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, box_info.category, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

        # Convert from OpenCV’s BGR format → Matplotlib’s RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"Frame {frame_id}")
        plt.show()


def load_video_annot(video_annot):
    """
    Load video-level annotations (i.e., clip-to-category mapping).

    Args:
        video_annot (str): Path to the video-level annotation file (e.g., "annotations.txt").

    Returns:
        dict[str, str]: Mapping between clip directory name and its action category.
    """
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            # Each line: "<clip_name>.jpg  <category>"
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    """
    Load the full Volleyball dataset (videos + tracking annotations).

    Args:
        videos_root (str): Path to the root directory containing all video folders.
        annot_root (str): Path to the root directory containing all tracking annotations.

    Returns:
        dict: Nested dictionary structure:
            {
                video_dir: {
                    clip_dir: {
                        'category': <clip category>,
                        'frame_boxes_dct': {frame_id: [BoxInfo, ...]}
                    }
                }
            }
    """
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}

    # Iterate over each video directory
    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        # Skip non-directory entries
        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        # Load annotations for this video (mapping from clip → category)
        video_annot = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        # Iterate over each clip in the video
        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            # Ensure this clip has a known category
            assert clip_dir in clip_category_dct

            # Load tracking annotations for this clip
            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            frame_boxes_dct = load_tracking_annot(annot_file)

            # Store both category and bounding box annotations
            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct
            }

        # Add video’s data to global dictionary
        videos_annot[video_dir] = clip_annot

    return videos_annot


def create_pkl_version():
    """
    Convert the full Volleyball dataset into a serialized pickle (.pkl) format.

    This step makes dataset loading much faster during training or evaluation.
    """
    annot_path = "/kaggle/working/annot_all.pkl"
    videos_root = f'{dataset_root}/volleyball_/videos'
    annot_root = f'{dataset_root}/volleyball_tracking_annotation/volleyball_tracking_annotation'

    # Build the hierarchical annotation structure
    videos_annot = load_volleyball_dataset(videos_root, annot_root)

    # Save to a .pkl file
    with open(annot_path, 'wb') as file:
        pickle.dump(videos_annot, file)


def test_pkl_version():
    """
    Load and test the saved .pkl annotation file to ensure correct structure.
    """
    pkl_path = '/kaggle/working/annot_all.pkl'

    with open(pkl_path, 'rb') as file:
        videos_annot = pickle.load(file)

    # Example: access specific clip and frame boxes
    boxes: List[BoxInfo] = videos_annot['0']['13456']['frame_boxes_dct'][13455]
    print(boxes[6].category)
    print(boxes[6].box)


if __name__ == '__main__':
    # Example usages (uncomment as needed):

    # Visualize one clip’s tracking annotations
    # annot_file = f'{dataset_root}/volleyball_tracking_annotation/volleyball_tracking_annotation/4/24745/24745.txt'
    # clip_dir_path = os.path.dirname(annot_file).replace(
    #     'volleyball_tracking_annotation/volleyball_tracking_annotation',
    #     'volleyball_/videos'
    # )
    # vis_clip(annot_file, clip_dir_path)

    # Create the pickle dataset
    # create_pkl_version()

    # Test the created dataset
    test_pkl_version()