
def two_step_prediction(final_prediction_date):

    if type(final_prediction_date) == str:
        final_prediction_date = datetime.datetime.strptime(final_prediction_date, "%Y-%m-%d")

    first_prediction_date = final_prediction_date - datetime.timedelta(days=7)
    __, pred1_diff, __, pred1 = run_prediction(pred_date=first_prediction_date, prediction_window=1)
    pred1_raw = pd.DataFrame(pd.concat([pred1[cn.Y_M_UNDIF], pred1[cn.Y_NM_UNDIF]]))
    pred1_diff = pred1_diff.T.set_index(pred1_diff.columns)

    pred1_combined = pred1_diff.join(pred1_raw, how='left')
    pred1_combined['pred1_final'] = (pred1_combined.sum(axis=1)).astype(int)
    # pred1_combined['pred1_final'] = [0 if x < 0 else x for x in pred1_combined['pred1_final']]

    __, pred2_diff, __, __ = run_prediction(pred_date=first_prediction_date, prediction_window=2)
    pred2_diff = pred2_diff.T.set_index(pred2_diff.columns)

    pred2_combined = pred2_diff.join(pred1_combined['pred1_final'], how='left')
    pred2_combined['pred2_final'] = (pred2_combined.sum(axis=1)).astype(int)
    # pred2_combined['pred2_final'] = [0 if x < 0 else x for x in pred2_combined['pred2_final']]

    return pred1_combined['pred1_final'].rename(first_prediction_date), \
           pred2_combined['pred2_final'].rename(final_prediction_date)


def batch_2step_prediction(prediction_dates):
    all_predictions_1stp = pd.DataFrame([])
    all_predictions_2stp = pd.DataFrame([])

    for dt in prediction_dates:
        _1step, _2step = two_step_prediction(final_prediction_date=dt)
        all_predictions_1stp = pd.concat([all_predictions_1stp, _1step], axis=1)
        all_predictions_2stp = pd.concat([all_predictions_2stp, _2step], axis=1)

    return all_predictions_1stp.T, all_predictions_2stp.T

    prediction_dates = pd.DataFrame(pd.date_range('2020-07-01', periods=9, freq='W-MON').astype(str), columns=[cn.FIRST_DOW])
    all_1step, all_2step = batch_2step_prediction(prediction_dates=prediction_dates[cn.FIRST_DOW])

    all_1step.index.rename(cn.FIRST_DOW, inplace=True)
    all_2step.index.rename(cn.FIRST_DOW, inplace=True)

    gf.save_to_csv(all_1step, file_name="1step_predictions", folder=fm.SAVE_LOC)
    gf.save_to_csv(all_2step, file_name="2step_predictions", folder=fm.SAVE_LOC)

