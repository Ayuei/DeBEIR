from typing import Dict, List
import numpy as np


def average_of_list_of_dicts(runs: List[Dict]):
    c_sum = 0.0

    for run in runs:
        c_sum += average_of_dict(run)

    return c_sum/len(runs)


def average_of_dict(run: Dict, ignore_nans=False):
    if ignore_nans:
        vals = []
        for value in run.values():
            try:
                int(value)
                vals.append(value)
            except ValueError:
                pass

        return sum(vals)/len(vals)

    return sum(run.values())/len(run)


scaling_factor = {
    'use0': [11, 12, 15, 16, 17, 25, 29, 34, 43, 48],
    'use1': [1,9,13,20,22,30,32,40,41,46],
    'use2': [2,18,19,23,31,36,37,39,47,49],
    'use3': [3,6,8,14,27,28,33,38,44,45],
    'use4': [4,5,7,10,21,24,26,35,42,50],
}

def calculate_scaling_factor(df, fold: int, metric='bpref@1000'):
    counts = 0

    for _, row in df.iterrows():
        if f'use{fold}' in row.variable_1 and metric == row.variable_0:
            if row.topic_num in scaling_factor[f'use{fold}']:
                if pd.isnull(row.value):
                    print(row.value)
                    counts +=1
    length = len(scaling_factor[f'use{fold}'])

    return (length-counts)/length
