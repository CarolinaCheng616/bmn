import numpy as np
import sys
import os.path as osp
import os
import json
import pickle
from scipy import interpolate
import math
from collections import defaultdict
import copy


def gen_duration():
    """
    1. read file mete = 'features/mete.json'
    2. get each video name and its subset(e.g. 'training') in mete
    3. for each video, read its pickle file in 'features/subset_image_resnet50_feature/name.pkl' and get its duration
    e.g.  for video 119914000, pkl file is 'features/train_image_resnet50_feature/119914000.pkl'
    4. write meta.json for each subset including videos' name, duration, annotations
    three json file: 'features/train_meta.json', 'features/val_meta.json', 'features/test_meta.json'
    """
    path = sys.path[0]
    path = '/'.join(path.split('/')[:-2]+['features'])
    with open(osp.join(path, 'mete.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
        # print(meta.keys())   # version, database
    train_image_path = osp.join(path, 'train_image_resnet50_feature')
    # train_audio_path = osp.join(path, 'train_audio_feature')
    val_image_path = osp.join(path, 'val_image_resnet50_feature')
    # val_audio_path = osp.join(path, 'val_audio_feature')
    test_image_path = osp.join(path, 'test_image_resnet50_feature')
    # test_audio_path = osp.join(path, 'test_audio_feature')
    database = meta['database']
    video_names = list(database.keys())
    # print(len(video_names))   # 1499
    train_video_infos = dict()
    val_video_infos = dict()
    test_video_infos = dict()
    train_num = 0
    val_num = 0
    test_num = 0
    for video_name in video_names:
        video_info = database[video_name]
        subset = video_info['subset']
        anno = video_info['annotations']
        if subset == 'training':
            train_num += 1
            file = open(osp.join(train_image_path, video_name+'.pkl'), 'rb')
        elif subset == 'validation':
            val_num += 1
            file = open(osp.join(val_image_path, video_name + '.pkl'), 'rb')
        else:
            test_num += 1
            file = open(osp.join(test_image_path, video_name + '.pkl'), 'rb')
        second = pickle.load(file, encoding='bytes').shape[0]
        file.close()
        new_video_info = dict()
        new_video_info['duration_second'] = second
        new_video_info['annotations'] = anno
        if subset == 'training':
            train_video_infos[video_name] = new_video_info
        elif subset == 'validation':
            val_video_infos[video_name] = new_video_info
        elif subset == 'testing':
            test_video_infos[video_name] = new_video_info
    train_meta = osp.join(path, 'train_meta.json')
    val_meta = osp.join(path, 'val_meta.json')
    test_meta = osp.join(path, 'test_meta.json')
    with open(train_meta, 'w', encoding='utf-8') as f:
        json.dump(train_video_infos, f, ensure_ascii=False, indent=2)
    with open(val_meta, 'w', encoding='utf-8') as f:
        json.dump(val_video_infos, f, ensure_ascii=False, indent=2)
    with open(test_meta, 'w', encoding='utf-8') as f:
        json.dump(test_video_infos, f, ensure_ascii=False, indent=2)
    # print(train_num)   # 1262
    # print(val_num)     # 120
    # print(test_num)    # 117


def read_features(video_name, image_path, audio_path):
    """
    read image feature and audio feature for a single video

    Args:
        video_name:
        image_path: video's image features directory
        audio_path: video's audio features directory

    Returns:
        concatenated image features and audio features of the video

    """
    image_file = osp.join(image_path, video_name + '.pkl')
    audio_file = osp.join(audio_path, video_name + '.pkl')
    with open(image_file, 'rb') as f:
        image_data = pickle.load(f, encoding='bytes')
    with open(audio_file, 'rb') as f:
        try:
            audio_data = pickle.load(f, encoding='bytes')
        except EOFError:
            audio_data = np.zeros(image_data.shape)
    length = min(image_data.shape[0], audio_data.shape[0])
    data = np.concatenate([image_data[:length], audio_data[:length]], axis=1)
    if data.shape[1] < 4096:
        data = np.pad(data, ((0, 0), (0, 4096-data.shape[1])), 'constant', constant_values=(0, 0))
    elif data.shape[1] > 4096:
        data = data[:, :4096]
    # length, 4096
    return data


def pool_features(features, temporal_length=100):
    """
    pooling features from its original temporal length to temporal_length for a single video by linear interpolate
    Args:
        features: concatenated image and audio features of a single video
        temporal_length: target temporal length

    Returns:
        pooled features at target temporal length
    """
    shape = features.shape
    interval = shape[0] / temporal_length
    ot = np.array(range(shape[0])) + 0.5
    min_time, max_time = min(ot), max(ot)
    ft = np.array([i * interval + 0.5 * interval for i in range(temporal_length)])
    f = interpolate.interp1d(ot, features, axis=0)
    zeros = np.zeros(shape[1])
    new_features = []
    for time in ft:
        if time < min_time or time > max_time:
            new_features.append(zeros)
        else:
            new_features.append(f(time))
    return np.stack(new_features)


def rescale_temporal_length(temporal_length=100):
    """
    rescale temporal length to target length for all videos of different subsets
    Args:
        temporal_length:  target temporal length

    generate pooled features for all videos by different subset
    'features/train_mean_100/xxx.pkl', 'features/val_mean_100/xxx.pkl', 'features/test_mean_100/xxx.pkl'
    """
    path = sys.path[0]
    path = '/'.join(path.split('/')[:-2] + ['features'])

    train_image_path = osp.join(path, 'train_image_resnet50_feature')
    train_audio_path = osp.join(path, 'train_audio_feature')
    val_image_path = osp.join(path, 'val_image_resnet50_feature')
    val_audio_path = osp.join(path, 'val_audio_feature')
    test_image_path = osp.join(path, 'test_image_resnet50_feature')
    test_audio_path = osp.join(path, 'test_audio_feature')

    train_meta_path = osp.join(path, 'train_meta.json')
    val_meta_path = osp.join(path, 'val_meta.json')
    test_meta_path = osp.join(path, 'test_meta.json')

    subsets = list()
    subsets.append(dict(name='train', image=train_image_path, audio=train_audio_path, meta=train_meta_path))
    subsets.append(dict(name='val', image=val_image_path, audio=val_audio_path, meta=val_meta_path))
    subsets.append(dict(name='test', image=test_image_path, audio=test_audio_path, meta=test_meta_path))

    for subset in subsets:
        name = subset['name']
        dire = osp.join(path, f'{name}_mean_{temporal_length}')
        if not os.path.exists(dire):
            os.makedirs(dire)
        image_path = subset['image']
        audio_path = subset['audio']
        meta_path = subset['meta']
        with open(meta_path, 'r', encoding='utf-8') as f:
            video_infos = json.load(f)
        video_names = list(video_infos.keys())
        for video_name in video_names:
            features = read_features(video_name, image_path, audio_path)
            pooled_features = pool_features(features, temporal_length)
            with open(f"{osp.join(dire, video_name+'.pkl')}", 'wb') as new_pkl:
                pickle.dump(pooled_features.astype(np.float32), new_pkl)


def truncate(temporal_length=2000, clips=10):
    """
    truncate a whole video into argument{clips} clips for all videos, including training, validation and testing

    Args:
        temporal_length:
        clips:

    Returns:

    """
    path = sys.path[0]
    path = '/'.join(path.split('/')[:-2] + ['features'])
    train_path = osp.join(path, f'train_mean_{temporal_length}')
    val_path = osp.join(path, f'val_mean_{temporal_length}')
    test_path = osp.join(path, f'test_mean_{temporal_length}')

    train_meta_path = osp.join(path, f'train_meta.json')
    val_meta_path = osp.join(path, f'val_meta.json')
    test_meta_path = osp.join(path, f'test_meta.json')

    subsets = list()
    subsets.append(dict(name='train', root=path, feature_path=train_path, meta_path=train_meta_path))
    subsets.append(dict(name='val', root=path, feature_path=val_path, meta_path=val_meta_path))
    subsets.append(dict(name='test', root=path, feature_path=test_path, meta_path=test_meta_path))

    length_per_clip = temporal_length // clips
    for subset in subsets:
        name, root, feature_path, meta_path = subset['name'], subset['root'], \
                                              subset['feature_path'], subset['meta_path']
        new_meta_dict = dict()
        new_meta_name = osp.join(root, f'{name}_meta_{clips}.json')
        truncate_feature_dir = osp.join(root, f'{name}_mean_{temporal_length}_{clips}')
        if not osp.exists(truncate_feature_dir):
            os.makedirs(truncate_feature_dir)
        with open(meta_path, 'r', encoding='utf-8') as f:
            video_infos = json.load(f)
        for video_name in video_infos:
            video_info = video_infos[video_name]
            duration = video_info['duration_second']
            duration_per_clip = duration / clips
            duration_list = [(duration_per_clip * i, duration_per_clip * (i + 1)) for i in range(clips)]
            meta_dict = defaultdict(list)
            idx = 0
            video_info['annotations'] = sorted(video_info['annotations'], key=lambda x: x['segment'][0])
            video_infos[video_name] = video_info
            for anno in video_info['annotations']:
                start, end = anno['segment']
                while idx < clips and start > duration_list[idx][1]:
                    idx += 1
                if idx == clips:
                    continue
                temp_idx = idx
                while temp_idx < clips and end > duration_list[temp_idx][1]:
                    temp_idx += 1
                if temp_idx == clips:
                    end = duration_list[-1][1]
                    temp_idx = clips - 1
                for i in range(idx, temp_idx):
                    meta_dict[i].append(dict(segment=[start - duration_per_clip * i,
                                                      duration_list[i][1] - duration_per_clip * i]))
                    start = duration_list[i][1]
                meta_dict[temp_idx].append(dict(segment=[start - duration_per_clip * temp_idx,
                                                         end - duration_per_clip * temp_idx]))
            feature_duration_list = [f'{length_per_clip * i}_{length_per_clip * (i + 1)}' for i in range(clips)]
            with open(osp.join(feature_path, video_name + '.pkl'), 'rb') as bf:
                features = pickle.load(bf, encoding='bytes')
            for i, feature_duration in enumerate(feature_duration_list):
                if meta_dict[i]:
                    new_meta_dict[f'{video_name}_{feature_duration}'] = dict(duration_second=duration_per_clip,
                                                                             annotations=meta_dict[i])
                with open(osp.join(truncate_feature_dir, f'{video_name}_{feature_duration}.pkl'), 'wb') as bf:
                    pickle.dump(features[length_per_clip * i: length_per_clip * (i + 1)], bf)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(video_infos, f, ensure_ascii=False, indent=2)
        with open(new_meta_name, 'w', encoding='utf-8') as f:
            json.dump(new_meta_dict, f, ensure_ascii=False, indent=2)


def joint_clip_by_video(video_info):
    """
    joint annotations of a single video to a whole annotation
    denote two annotations as anno1={start1, end1}, anno2 = {start2, end2}
    if abs(end1-start2) < DISTANCE, then joint them to a single annotation

    process annotations on a single video
    """
    start_list = sorted([[proposal['segment'][0], proposal['segment'][1], [proposal['score']], i]
                         for i, proposal in enumerate(video_info)], key=lambda x: x[0])
    end_list = sorted([[proposal['segment'][0], proposal['segment'][1], [proposal['score']], i]
                       for i, proposal in enumerate(video_info)], key=lambda x: x[1])
    vidx2start = dict([(sp[3], i) for i, sp in enumerate(start_list)])
    vidx2end = dict([(sp[3], i) for i, sp in enumerate(end_list)])
    start_idx, end_idx = 0, 0
    DISTANCE = 1   # max interval between the start point and end point
    new_idx = len(start_list)
    while start_idx < len(start_list):
        start_point = start_list[start_idx]
        start, end_, score_list1, i1 = start_point
        start_, end, score_list2, i2 = end_list[end_idx]
        while (start - end) > DISTANCE:
            end_idx += 1
            start_, end, score_list2, i2 = end_list[end_idx]
        if i1 != i2 and abs(start - end) <= DISTANCE:
            # merge
            new_segment = [start_, end_, score_list1 + score_list2, new_idx]
            sidx1, sidx2 = vidx2start[i1], vidx2start[i2]
            eidx1, eidx2 = vidx2end[i1], vidx2end[i2]
            start_list[sidx2] = list(copy.deepcopy(new_segment))
            end_list[eidx1] = list(copy.deepcopy(new_segment))
            del start_list[start_idx]  # start_idx = sidx1
            del end_list[end_idx]      # end_idx = eidx2
            vidx2start = dict([(sp[3], i) for i, sp in enumerate(start_list)])
            vidx2end = dict([(sp[3], i) for i, sp in enumerate(end_list)])
            new_idx += 1
            start_idx -= 1
        start_idx += 1

    new_video_info = list()
    for sp in start_list:
        new_video_info.append(dict(score=float(sum(sp[2])/len(sp[2])), segment=[sp[0], sp[1]]))
    return new_video_info


def joint_clip():
    """
    joint video clip annotations to the whole video annotations
    and write file

    """
    path = sys.path[0]
    path = '/'.join(path.split('/')[:-2] + ['features'])
    # whole video: 219131500
    # partial video: 219131500_0_200
    # read duration infos for each whole video
    anno_path = osp.join(path, 'val_meta.json')
    video_duration_per_clip = dict()
    with open(anno_path, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    whole_video_names = list(videos.keys())
    for video_name in whole_video_names:
        video_duration_per_clip[video_name] = videos[video_name]['duration_second'] / 10
    # get predicted proposals for each partial video
    origin_results = osp.join(path, 'results_10.json')
    with open(origin_results) as f:
        video_infos = json.load(f)
    partial_video_names = list(video_infos.keys())

    whole_video_infos = defaultdict(list)
    for video_name in partial_video_names:
        video_name_split = video_name.split('_')
        whole_video_name, idx = video_name_split[0], int(video_name_split[1]) // 200
        partial_video_info = video_infos[video_name]
        duration_per_clip = video_duration_per_clip[whole_video_name]
        for video_info in partial_video_info:
            start = float(video_info['segment'][0]) + idx * duration_per_clip
            end = float(video_info['segment'][1]) + idx * duration_per_clip
            whole_video_infos[whole_video_name].append(dict(score=video_info['score'], segment=[start, end]))

    results_path = osp.join(path, 'results_split.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(whole_video_infos, f, ensure_ascii=False, indent=2)

    for video_name in whole_video_infos:
        whole_video_infos[video_name] = joint_clip_by_video(whole_video_infos[video_name])

    new_results_path = osp.join(path, 'results_joint.json')
    with open(new_results_path, 'w', encoding='utf-8') as f:
        json.dump(whole_video_infos, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    joint_clip()
