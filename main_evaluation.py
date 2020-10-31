from prediction import general_purpose_functions as gf
from prediction import file_management as fm
from prediction import column_names as cn
import pandas as pd

if __name__ == '__main__':
    results = gf.read_pkl(file_name='test_result_bs_20201031_1257.p',
                          data_loc=fm.SAVE_LOC)

    for i in range(0, len(results)):

        if i == 0:
            all_dicts = results[i]
        else:
            all_dicts.update(results[i])

    all_predictions = pd.DataFrame([])
    for k in all_dicts.keys():
        _preds = all_dicts[k]['all_predictions']

        all_predictions = pd.concat([all_predictions, _preds])


