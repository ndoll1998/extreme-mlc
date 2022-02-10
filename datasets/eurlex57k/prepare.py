import os
import json
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict

def build_dataframe(data_dir:str):
    
    columns = defaultdict(list)
    for fname in tqdm(os.listdir(data_dir), desc=data_dir):
        # open file and load content
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'r') as f:
            data = json.loads(f.read())
    
        # write data into columns
        columns['ID'].append(data['celex_id'])
        columns['main_body'].append(' '.join(data['main_body']).replace('\n', ' '))
        columns['labels'].append(' '.join(data['concepts']).replace('\n', ' '))
        for key in ['title', 'header', 'recitals']:
            columns[key].append(data[key]) 

    # convert to pandas dataframe
    index = columns.pop('ID')
    return pd.DataFrame(data=columns, index=index)

if __name__ == '__main__':

    # build dataframes
    build_dataframe("./train").to_csv("train.csv")
    build_dataframe("./dev").to_csv("dev.csv")
    build_dataframe("./test").to_csv("test.csv")
    

