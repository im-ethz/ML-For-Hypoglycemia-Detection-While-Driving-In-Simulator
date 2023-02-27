import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn import metrics

score_functions = {
    'AUROC': lambda y_true, y_pred_proba, _: metrics.roc_auc_score(y_true, y_pred_proba),
    'AUPRC': lambda y_true, y_pred_proba, _: metrics.average_precision_score(y_true, y_pred_proba),
    'BACC': lambda y_true, y_pred_proba, threshold: metrics.balanced_accuracy_score(y_true, y_pred_proba > threshold),
    'F1': lambda y_true, y_pred_proba, threshold: metrics.f1_score(y_true, y_pred_proba > threshold),
    'MCC': lambda y_true, y_pred_proba, threshold: metrics.matthews_corrcoef(y_true, y_pred_proba > threshold),
    'Sensitivity': lambda y_true, y_pred_proba, threshold: metrics.recall_score(y_true, y_pred_proba > threshold, pos_label=1),
    'Specificity': lambda y_true, y_pred_proba, threshold: metrics.recall_score(y_true, y_pred_proba > threshold, pos_label=0),
    'test_pos': lambda y_true, *_: np.sum(y_true == 1),
    'test_neg': lambda y_true, *_: np.sum(y_true == 0),
}


def print_performance(df: pd.DataFrame, name: str = 'results', print_df=True, print_csv=False, print_sample_counts=True):
    df_print = df.agg(lambda x: f'{x.mean():.2f}±{x.std(ddof=0):.2f}')
    df_print.index = pd.MultiIndex.from_tuples(df_print.index)
    df_print = df_print.unstack(level=0)
    df_print.name = name
    df_print.index.name = name
    print_cols = ['AUROC', 'AUPRC', 'BACC', 'F1', 'MCC', 'Sensitivity', 'Specificity'] + (['test_pos', 'test_neg'] if print_sample_counts else [])
    df_print = df_print[print_cols]
    if print_df:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            print(df_print.reset_index().to_string(index=False))
    if print_csv:
        import sys
        df_print.to_csv(sys.stdout)
    return df_print, df

def evaluate_train_performance(y_true, y_pred, groups, envs, threshold=-1):
    scores = []
    for group in sorted(np.unique(groups)):
        score = {'id': group}
        y_trues, y_preds = y_true[groups == group], y_pred[groups == group]
        thresh = threshold if threshold != -1 else get_optimal_threshold(y_trues, y_preds)

        for key, f in score_functions.items():
            try:
                score[key] = {'All': f(y_trues, y_preds, thresh)}
            except Exception as e:
                print(f'Unable to compute {key} All for group {group}: {e}')
                score[key] = {'All': np.nan}

        for env in ['Highway', 'Rural', 'Town']:
            y_trues, y_preds = y_true[(groups == group) & (envs == env)], y_pred[(groups == group) & (envs == env)]

            for key, f in score_functions.items():
                try:
                    score[key][env] = f(y_trues, y_preds, thresh)
                except Exception as e:
                    print(f'Unable to compute {key} {env} for group {group}: {e}')
                    score[key][env] = np.nan

        scores.append(score)

    df = pd.DataFrame(scores)
    df.set_index('id', inplace=True)
    for col in df.columns:
        tmp = pd.json_normalize(df[col])
        target_cols = pd.MultiIndex.from_product([[col], tmp.columns])
        df[target_cols] = tmp.values
        df.drop(columns=[col], inplace=True)

    return df.mean()

def evaluate_performance(y_true, y_pred, groups, envs, print_sample_counts=False, print_df=True, print_csv=False, print_test_sample_counts=True, threshold=-1, name='results'):
    if print_sample_counts:
        sample_counts = {}

        def get_sample_counts(labels_test,):
            counts = {
                'test_all': len(labels_test),
                'test_neg': np.sum(labels_test == 0),
                'test_pos': np.sum(labels_test == 1)
            }
            return counts

        sample_counts['All'] = get_sample_counts(y_true)
        for env in ['Highway', 'Rural', 'Town']:
            sample_counts[env] = get_sample_counts(y_true[envs == env])
        sample_counts = pd.DataFrame(sample_counts).transpose()
        sample_counts.name = 'Sample Counts'
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            print(sample_counts)

    scores = []
    for group in sorted(np.unique(groups)):
        score = {'id': group}
        y_trues, y_preds = y_true[groups == group], y_pred[groups == group]
        thresh = threshold if threshold != -1 else get_optimal_threshold(y_trues, y_preds)

        for key, f in score_functions.items():
            try:
                score[key] = {'All': f(y_trues, y_preds, thresh)}
            except Exception as e:
                print(f'Unable to compute {key} All for group {group}: {e}')
                score[key] = {'All': np.nan}

        for env in ['Highway', 'Rural', 'Town']:
            y_trues, y_preds = y_true[(groups == group) & (envs == env)], y_pred[(groups == group) & (envs == env)]

            for key, f in score_functions.items():
                try:
                    score[key][env] = f(y_trues, y_preds, thresh)
                except Exception as e:
                    print(f'Unable to compute {key} {env} for group {group}: {e}')
                    score[key][env] = np.nan

        scores.append(score)

    df = pd.DataFrame(scores)
    df.set_index('id', inplace=True)
    for col in df.columns:
        tmp = pd.json_normalize(df[col])
        target_cols = pd.MultiIndex.from_product([[col], tmp.columns])
        df[target_cols] = tmp.values
        df.drop(columns=[col], inplace=True)

    return print_performance(df, name, print_df=print_df, print_csv=print_csv, print_sample_counts=print_test_sample_counts)


def get_optimal_threshold(y_trues, y_preds):
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
    return thresholds[np.argmax(tpr - fpr)]
