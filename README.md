# Evaluating Speech Signal Representations for Emotion Recognition from Speech
### CPSC 581 Final Project: Spring 2025
-----

## Getting Started

After cloning the repository, create a conda environment for Python 3 using the `requirements.txt` file:
```
conda create --name <env_name> --file requirements.txt
```
Activate the conda environment by running:
```
conda activate <env_name>
```
where `<env_name>` is your name of choice for the conda environment.

## Using this repo
The results for this project can be reproduced using the following steps:
1. Clone this repo, navigate into the repo, and set up the conda environment as directed above.
2. Download the followiing 8 data files and move them into the `data` directory.
   - mfcc_emotion_x: https://drive.google.com/file/d/1yjyQTQFSIRUK-zKI7B9H8iJQnqHW_tdk/view?usp=sharing
   - mfcc_emotion_y: https://drive.google.com/file/d/1nJuFBv6rC8XhUqFEQhLUO3Ha2zdesfZL/view?usp=sharing
   - mfcc_intensity_x: https://drive.google.com/file/d/1--bZ3jOVJUHnDEl2B2nbi7Ya3vlQY4fB/view?usp=sharing
   - mfcc_intensity_y: https://drive.google.com/file/d/1-0LLLns_qvJqrbyBhO1IGabqiszGp_xF/view?usp=sharing
   - gfcc_emotion_x: https://drive.google.com/file/d/1-3nfuvFj48zimWQpEG2EOehpUvJO-yZj/view?usp=sharing
   - gfcc_emotion_y: https://drive.google.com/file/d/1-4IGR4ChYi4YSQ6I9F0R_gNkw4-p4DV4/view?usp=sharing
   - gfcc_intensity_x: https://drive.google.com/file/d/1-8QNZwkWUJJ-yW0kEXM6Jm2YZXwvYQM8/view?usp=sharing
   - gfcc_intensity_y: https://drive.google.com/file/d/1-BvPl_l-J730Hl-CPSGz9v2ZmeQ9oqA-/view?usp=sharing

   *Please note that for the purpose of this project, since obtaining the raw speech signals from RAVDESS requires signed approval/licensing, we provide only the already-processed speech signals.*
4. Activate the conda environment as directed above.
5. To train all the models using the hyperparameters specified in the final project report, run
   ```
   bash run.sh
   ```
   This will run experiments for all data (MFCC, GFCC) and task (emotion classification, intensity classification) settings.

   After each model is trained, a scikit-learn classification report and confusion matrix will be generated for the model to assess its performance on the train, validation, and test splits of the data. These results will be saved in a newly and automatically created `results` directory.
7. Once all models have completed training, aggregate the scores of performance for each data and task setting by running:
   ```
   python analyze.py
   ```
   This will create a subdirectory called `results/_summary` containing CSV files for each data and task setting that sort the model and hyperparameter settings by accuracy (descending order). These files can then be used to obtain final observations as reported in the project report.
