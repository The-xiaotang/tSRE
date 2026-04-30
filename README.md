# Table Sematic Relationship Extraction
- baseline
  - related paper: Annotating Columns with Pre-trained Language Models(Sigmod'2022)
  - train.py # baseline train code(paddle version)
  - infer.py # baseline infer code(paddle version)
- dataset
  - train.zip # train dataset
    - Train_Set
      - xxx.csv # xxx: label name
  - test.csv # test dataset
  - labels.csv # labels set
* environment
  ```
  python -m pip install -r requirements.txt
  ```
* run command
  ```
  python train.py
  python infer.py
  ```
