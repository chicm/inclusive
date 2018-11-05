# Inclusive Images Challenge

* Step 1: Configure local_settings.py:

  Set DATA_DIR to the directory containing all csvs and a sub dir named "stage_1_test_images".
  Set TRAIN_IMG_DIR to the open images train image file diretory.

* Step 2: Install required python packages:  
  pytorch 0.4.1, torchvision, attrdict, PIL

* Step 3: Run preprocess

  ```python3 preprocess.py```

* Step 4: Train models

  Train 2 models, each model take around 10 days on a single P100 GPU, the train_single_class.py training step can be stopped when it get around 0.94 top 10 accuracy. The train.py training step can be stopped when it get around 0.7 f2 score.
  ```
  python3 train_single_class.py --epochs 10 --batch_size 64 --iter_val 300
  python3 train_single_class.py --lrs cosine --lr 0.001 --min_lr 0.0001 --epochs 1000 --batch_size 64 --iter_val 300
  python3 train.py --epochs 10 --end_index 1000 --init_ckp <model file generated by above step> --init_num_classes 7172 --batch_size 64
  python3 train.py --lrs cosine --lrs cosine --lr 0.0005 --min_lr 0.00005 --epochs 100 --end_index 1000 --batch_size 64
  ```

  ```
  python3 train_single_class.py --epochs 10 --backbone resnet101 --batch_size 64 --iter_val 300
  python3 train_single_class.py --lrs cosine --lr 0.001 --min_lr 0.0001 --epochs 1000 --backbone resnet101 --batch_size 64 --iter_val 300
  python3 train.py --lrs cosine --epochs 10 --end_index 1000 --init_ckp <model file generated by above step> --init_num_classes 7172 --backbone resnet101 --batch_size 64
  python3 train.py --lrs cosine --lrs cosine --lr 0.0005 --min_lr 0.00005 --epochs 100 --end_index 1000 --backbone resnet101 --batch_size 64
  ```


* Step 5: Make Stage 1 prediction and submission

  ```
  python3 predict.py --end_index 1000 --sub_file sub1.csv --th 0.04
  python3 predict.py --end_index 1000 --sub_file sub2.csv --th 0.04 --backbone resnet101
  python3 predict.py --ensemble_np <np files generated by above 2 predictions, splitted by ','> --th 0.04 --sub_file ensemble.csv
  ```
  ensemble.csv is the stage 1 submission file.

* Step 6: Make Stage 2 prediction and submission

  Open settings.py, set TEST_IMG_DIR to directory of stage 2 image directory. And set the value of STAGE_1_SAMPLE_SUBMISSION to the file path of stage 2 sample submission file.

  ```
  python3 predict.py --end_index 1000 --sub_file sub1.csv --th 0.04
  python3 predict.py --end_index 1000 --sub_file sub2.csv --th 0.04 --backbone resnet101
  python3 predict.py --ensemble_np <np files generated by above 2 predictions, splitted by ','> --th 0.04 --sub_file ensemble.csv
  ```
  ensemble.csv is the stage 2 submission file.
