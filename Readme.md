# SeC-GAN - Generative Adversarial Network for Just-in-Time Defect Prediction with Semantic Changes Sensitivity

## I'm grateful that you're interested in my work! :)

![Alt Text](https://c.tenor.com/OXvCPNJKJB8AAAAC/yes-im-so-excited.gif)

I hope the following guide will make it easier for you to understand what we did.

## Documentation

There is documentation for all code [here](https://shirshir05.github.io/SeC_GAN__Just_in_Time_Defect_Prediction.github.io-/), and we recommend you look.

**For example**: 

![Alt Text](https://github.com/shirshir05/SeC_GAN__Just_in_Time_Defect_Prediction.github.io-/blob/master/Image/documentation%20.jpg)



## Requirements

Install all requirements in the "requirements.txt" file


## Step 1 - Extract Data:

1. Do clone to [bic (bug induce commit) project] (https://github.com/amir9979/bic).
2. Requirements: 
    * Python 3.9 - then run: 
    ```
      python -m pip install --upgrade pip
      pip install pytest 
      pip install gitpython
      pip install jira
      pip install termcolor 
      pip install openpyxl  
      pip install javalang
      pip install pathlib
      pip install junitparser
      pip install pandas
      pip install numpy
      pip install pydriller
      pip install dataclasses
      pip install jsons
      python ./javadiff/setup.py install
     ```
    * java version 11 and 8
3. Do clone to project that you want to extract data to directory name "local_repo" the repository. For example: apache/mahout.
4. Execute: 
```
python Main.py [0]
```
  * Note: [0]  -indicate the number of commit extracts in a window size of 50.  For large project set [0, 1, 2.....200].
 5. After the run the file save in "./results/" folder. 
 6. To merge all file run:
  ```
    import pandas as pd
    import glob
    all_files = glob.glob("./results/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    pd.concat(li, axis=0, ignore_index=True).to_csv('./results/all.csv', index=False
   ```
 7. Now, all the data save in './results/all.csv'.
    

## Step 2 - Preprocessing and SZZ algorithm

1. In Data folder open for each project directory with the name NAME_PROJECT (e.g mahout) and put the file "all.csv" in the directory. For example: Data\mahout\all.csv). 
2. In "variable.py" add the NAME_PROJECT and the key_issue (according to JIRA) to function get_key_issue().
3. Update NAME_PROJECT in list projects (namly "projects") and run :
   ```
   python CreateData.py
   ```
5. After the run train, validation and test file created in the path Data\NAME_PROJECT\train_test. In addition, a folder named "blame" will be created with the results of the SZZ algorithm.

## Step 3 -  [Train CTGAN](https://github.com/sdv-dev/CTGAN)

1. Train CTGAN algorithm in folder CTGAN -> "run.py".
   ```
   python run.py NAME_PROJECT
   ```
     * Note: NAME_PROJECT -indicate the project name as saved in the Data folder.
3. You can change the parameters() function with another parameters.
4. After this process update in (Data -> NAME_PROJECT -> CTGAN -> FILE_CREATE) the FILE_CREATE from this runing.

## Step 4 -  Generate Fake Data

1. Generate fake example using the generators from step 3. 
   ```
   python GenerateFakeData.py 
   ```
     * Note: Update the NAME_PROJECT in list projects.
## Step 5 - Run Algoritham 
1. You can run each algorithm in file Main_Algorithm.
2. Update NAME_PROJECT in list projects.
3. Execute:
   ```
   python Main_Algorithm.py path name_model with_G
   ```
   * Note: 
      path (int - save the number of runing)
      name_model (RF, LR or D), w
      with_G( 0 (False), 1 (True)).


## Good luck !!! 
