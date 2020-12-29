### There are modified files in mmaction2:

loading.py: mmaction/datasets/pipelines/loading.py
trunet_dataset.py: mmaction/datasets/trunet_dataset.py
eval_detection.py: mmaction/core/evaluation/eval_detection.py
bmn_200x4096x10_4x32_70e_trunet_truncate_feature.py: configs/localization/bmn/bmn_200x4096x10_4x32_70e_trunet_truncate_feature.py
report_trunet_map.py: tools/analysis/report_trunet_map.py



### my data process file:

data_process.py: pre-process and post-process trunet dataset features.



### annotation and result files

val_meta.json: original validation set annotations

val_meta_10.json: validation set annotations splitted into 10 clips for every video

results_10.json: results of splitted validation set (clip results)

results_joint.json: jointed results (whole video results)