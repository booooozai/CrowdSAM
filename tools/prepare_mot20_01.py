import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def filter_coco_by_image_ids(coco_data, image_ids):
    image_ids = set(image_ids)
    images = [img for img in coco_data['images'] if int(img['id']) in image_ids]
    annotations = [ann for ann in coco_data['annotations'] if int(ann['image_id']) in image_ids]
    out = {
        'images': images,
        'annotations': annotations,
        'videos': coco_data.get('videos', []),
        'categories': coco_data['categories'],
    }
    return out


def copy_mot20_01(src_root, dst_root):
    src_seq = src_root / 'train' / 'MOT20-01'
    dst_seq = dst_root / 'train' / 'MOT20-01'
    dst_seq.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src_seq / 'img1', dst_seq / 'img1', dirs_exist_ok=True)
    shutil.copy2(src_seq / 'seqinfo.ini', dst_seq / 'seqinfo.ini')

    dst_gt = dst_seq / 'gt'
    dst_gt.mkdir(exist_ok=True)
    shutil.copy2(src_seq / 'gt' / 'gt.txt', dst_gt / 'gt.txt')


def main():
    parser = argparse.ArgumentParser(description='Prepare MOT20-01 10-shot split for CrowdSAM.')
    parser.add_argument(
        '--src_root',
        default='/share/Pub_Datasets/chenshengbo/mot20/MOT20',
        help='Original MOT20 root.',
    )
    parser.add_argument(
        '--dst_root',
        default='dataset/mot20',
        help='Project-local MOT20 output root.',
    )
    parser.add_argument('--n_shot', type=int, default=10)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    annotation_path = src_root / 'annotations' / 'mot20_01_train.json'
    out_ann_dir = dst_root / 'annotations'
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    copy_mot20_01(src_root, dst_root)

    data = json.load(open(annotation_path))
    image_ids = sorted(int(img['id']) for img in data['images'])
    if args.n_shot <= 0 or args.n_shot >= len(image_ids):
        raise ValueError(f'n_shot must be in [1, {len(image_ids) - 1}], got {args.n_shot}')

    sample_positions = np.linspace(0, len(image_ids) - 1, args.n_shot)
    train_ids = sorted({image_ids[int(round(pos))] for pos in sample_positions})
    if len(train_ids) != args.n_shot:
        raise RuntimeError(f'Expected {args.n_shot} unique train ids, got {len(train_ids)}')
    val_ids = [image_id for image_id in image_ids if image_id not in set(train_ids)]

    train_data = filter_coco_by_image_ids(data, train_ids)
    val_data = filter_coco_by_image_ids(data, val_ids)

    train_path = out_ann_dir / 'mot20_01_10shot_train.json'
    val_path = out_ann_dir / 'mot20_01_10shot_val.json'
    json.dump(train_data, open(train_path, 'w'), ensure_ascii=True)
    json.dump(val_data, open(val_path, 'w'), ensure_ascii=True)

    print(f'Copied MOT20-01 to {dst_root / "train" / "MOT20-01"}')
    print(f'Wrote {train_path}: {len(train_data["images"])} images, {len(train_data["annotations"])} annotations')
    print(f'Wrote {val_path}: {len(val_data["images"])} images, {len(val_data["annotations"])} annotations')
    print(f'Train image ids: {train_ids}')


if __name__ == '__main__':
    main()
