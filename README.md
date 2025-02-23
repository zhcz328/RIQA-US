# RIQA-US


## :hammer: PostScript
&ensp; :smile: This project is the pytorch implementation of RIQA-US;

&ensp; :laughing: Our experimental platform is configured with <u>One *RTX3090 (cuda>=11.0)*</u>; 

&ensp; :blush: Currently, this code is available for our proposed dataset;  

<!-- &ensp; :smiley: For codes and assessment that related to dataset ***CardiacUDA***; -->


## :computer: Installation


1. You need to build the relevant environment first, please refer to : [**environment.yml**](environment.yml)

2. Install Environment:
    ```
    conda env create -f envs.yml
    ```

+ We recommend you to use Anaconda to establish an independent virtual environment, and python > = 3.8.3; 


## :blue_book: Data Preparation

### *1. dataset*
 * This project provides the use case of Ultrasound Image Quality Assessment task;

 * The hyperparameters setting of the dataset can be found in the **configs/xxx.yaml**, where you could do the parameters' modification;

   1. Download & Unzip the dataset.

      The ***dataset*** is composed as: /train & /val.

   2. The source code of loading the dataset exist in path :

      ```./utils/datasets.py```

   3. In **configs/xxx.yaml**, you can set the ***dataset_path***.

### *2. Dataset access*
  * Dataset access can be obtained by contacting (xxx) and asking for a license.
    
## :feet: Training

1. In this framework, after the parameters are configured in the file **configs/train.py** and **main.py** , you only need to use the command:

    ```shell
    python main.py
    ```


## :feet: Testing
1. Put the checkpoint file in ***./output*** directory.
2. Set the ***weights_path*** in ***./configs/evaluate.yaml***.
3. you only need to use the command:

   ```shell
    python evaluate.py
    ```
