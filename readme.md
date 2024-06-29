### 1. Construct dataset:
0. Download data that are not available via API calls as instructed in data/data_preparation_guide.md.

1. Prepare the complete table data by create a master data file with data from data file downloaded and Yahoo! Finance.
      ```
      python prepare_table_data.py
      ```
      The complete data could be found in file data/all_table_data.csv

2. Construct dataset by Pair the report with the corresponding tabular data and output as json file. The constructed data includes three subsets: 
      train/test/validation.  And process dataset for fine-tuning:

      ```
      python construct_dataset.py
      ```

### 2. Fine-tuning (optional):

1. Finetune model with DataTales:
      ```
      python finetune_llm.py
      ```


### 3. Inference:
 1. Run inference with DataTales test set:
      ```
    python inference.py 
      ```

    
### 4. Evaluation:
1. Reasoning evaluation:
   ```
    python reasoning_evaluation.py
   ```
2. Style analysis:
   ```
   python style_evaluation.py
   ```