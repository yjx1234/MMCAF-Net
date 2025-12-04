import pickle
import pandas as pd

class CTPE:
    def __init__(self, name, is_positive, parser, num_slice, first_appear, avg_bbox, last_appear):
        self.study_num = name
        self.is_positive = is_positive
        self.phase = parser
        self.num_slice = num_slice
        self.first_appear = first_appear
        self.bbox = avg_bbox
        self.last_appear = last_appear

    def __len__(self):
        return self.num_slice

#w all information
bbox_path = '/home/zcd/yujianxun/G_first_last_nor.csv'
pkl_path = '/home/zcd/yujianxun/series_list_last_AG.pkl'

df = pd.read_csv(bbox_path)
with open(pkl_path, 'wb') as pkl_file:
    all_ctpes = [CTPE(name = row['NewPatientID'], 
                    is_positive = row['label'],
                    parser = row['parser'],
                    num_slice = row['num_slice'],
                    first_appear = row['first_appear'],
                    avg_bbox = row['avg_bbox'],
                    last_appear = row['last_appear']) for index, row in df.iterrows()]
    pickle.dump(all_ctpes, pkl_file)

