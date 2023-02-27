import pickle

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn import pipeline, preprocessing, linear_model, neural_network
import xgboost

from helper import evaluate_performance, evaluate_train_performance, print_performance

np.random.seed(0)

############# USER SETTING

WINDOW_SIZE = 60

STUDY_1_SUBJECTS = [105, 106, 107, 108, 111, 113, 114, 115, 116, 117, 119, 120, 121, 123, 124, 125, 127, 129]
STUDY_2_SUBJECTS = [301, 302, 303, 304, 307, 310, 312, 313, 315]
#############

def get_model_pipeline() -> pipeline.Pipeline:
    pipe = pipeline.Pipeline([
        ('scale', preprocessing.StandardScaler()),
        ('predict', linear_model.LogisticRegression(C=2e-3, class_weight='balanced', n_jobs=8)),

        ### Robustness Checks
        #('predict', linear_model.LogisticRegression(C=2e-3, class_weight='balanced')), # Ridge
        #('predict', linear_model.LogisticRegression(C=2e-3, penalty='l1', solver='saga', class_weight='balanced')), # Lasso
        #('predict', linear_model.LogisticRegression(C=2e-3, l1_ratio=0.5, penalty='elasticnet', solver='saga', class_weight='balanced')), # Elasticnet
        #('predict', xgboost.sklearn.XGBClassifier(n_estimators=10, min_samples_split=10, max_depth=3, verbosity=0)), # XGBoost
        #('predict', neural_network.MLPClassifier(activation='logistic', max_iter=50)), # MLP
        #('predict', ensemble.GradientBoostingClassifier(n_estimators=10, min_samples_split=10, max_depth=3)), # Gradient boosting
    ])
    return pipe


def train_pipe(X_train: pd.DataFrame, y_train, groups_train, envs_train, X_test, y_test, groups_test, envs_test):
    prediction = np.zeros(len(X_test))
    coefs = []

    train_results = []

    pipe = get_model_pipeline()
    for subj in np.sort(np.unique(groups_test)):
        X_tr, y_tr = X_train[groups_train != subj], y_train[groups_train != subj]
        X_te, y_te = X_test[groups_test == subj], y_test[groups_test == subj]

        pipe.fit(X_tr, y_tr)
        try:
            coefs.append(pipe['predict'].coef_)
        except:
            coefs.append(None)
        prediction[groups_test == subj] = pipe.predict_proba(X_te)[:, 1]
        print('.', end='')

        train_result = evaluate_train_performance(y_tr, pipe.predict_proba(X_tr)[:, 1],
                                                  groups_train[groups_train != subj],
                                                  envs_train[groups_train != subj])
        train_results.append(train_result)

    print('')
    train_results = pd.concat(train_results, axis=1).transpose()
    print_performance(train_results, 'train_score', print_df=False, print_csv=True, print_sample_counts=False)
    return prediction, np.array(coefs)


def train_pipe_val(X_train: pd.DataFrame, y_train, groups_train, envs_train, X_test, y_test, groups_test, envs_test):
    prediction = np.zeros(len(X_test))
    coefs = []

    pipe = get_model_pipeline()
    pipe.fit(X_train, y_train)
    try:
        coefs.append(pipe['predict'].coef_)
    except:
        coefs.append(None)
    prediction = pipe.predict_proba(X_test)[:, 1]

    train_results = evaluate_train_performance(y_train, pipe.predict_proba(X_train)[:, 1], groups_train, envs_train)

    print('')
    print_performance(train_results.to_frame().transpose(), 'train_val_score', print_df=False, print_csv=True, print_sample_counts=False)
    return prediction, np.array(coefs)

def load_data(window_size_sec):
    filename = f'./data/allwitheye_participants_{window_size_sec:03d}.parquet'
    print(f'Loading overall data from file {filename}')
    df = pd.read_parquet(filename)

    return df


def run_full_eval(X, train_indices, test_indices, train_fun, feature_sets, label_column):
    predictions = {}
    all_coefs = {}
    all_stats = {}
    all_raw_stats = {}

    print(f'Train subjects (n = {len(X[train_indices]["subject_id"].unique())}): {sorted(X[train_indices]["subject_id"].unique())}')
    print(f'Test subjects (n = {len(X[test_indices]["subject_id"].unique())}): {sorted(X[test_indices]["subject_id"].unique())}')

    for desc, features in feature_sets.items():
        predictions_main, coefs = train_fun(X[train_indices][features], X[train_indices][label_column],
                                            X[train_indices]['subject_id'], X[train_indices]['env'],
                                            X[test_indices][features], X[test_indices][label_column],
                                            X[test_indices]['subject_id'], X[test_indices]['env']
                                            )

        stats, raw_stats = evaluate_performance(X[test_indices][label_column], predictions_main, X[test_indices]['subject_id'], X[test_indices]['env'],
                                                print_csv=True, print_sample_counts=False, threshold=-1, name=desc
                                                )

        all_coefs[desc] = {'coefs': coefs, 'features': features}

        try:
            pipe = get_model_pipeline()
            pipe.fit(X[train_indices][features], X[train_indices][label_column])
            all_coefs[desc]['train_coefs'] = pipe['predict'].coef_
        except:
            pass

        predictions[f'pred_{desc.lower()}'] = predictions_main
        all_stats[desc.lower()] = stats
        all_raw_stats[desc.lower()] = raw_stats

        print('=' * 80)
    return predictions, all_coefs, all_stats, all_raw_stats


def run_evaluation(window_size_sec):
    label_column = 'y_39'
    df = load_data(window_size_sec)

    # feature sets
    et_behavior_signals = ['v']
    et_behavior_agg = ['mean', 'std']
    et_behavior = [f'gaze+{agg}_{signal}_eye' for agg in et_behavior_agg for signal in et_behavior_signals]

    et_movement_signals = ['fixationduration']
    et_movement_agg = ['count', 'std_duration']
    et_movement = [f'{signal}+{agg}_eye' for agg in et_movement_agg for signal in et_movement_signals]

    et_features = et_movement + et_behavior

    can_signals = ['velocity', 'steer', 'brake', 'gas']
    can_agg = ['std', 'energy']
    can_features = [f'{signal}_{agg}_car' for agg in can_agg for signal in can_signals]

    feature_sets = {
        'CAN+ET': can_features + et_features,
        'CAN': can_features,
        'ET': et_features,
    }


    # RUN STUDY 1 (pronounced hypoglycemia) – LOSO
    print(f'{"=" * 80}\nRUN STUDY 1\n{"=" * 80}')
    train_function = train_pipe
    train_indices, test_indices = df['train'] & df['subject_id'].isin(STUDY_1_SUBJECTS), ~df['train'] & df['subject_id'].isin(STUDY_1_SUBJECTS)
    predictions, coefs, stats, raw_stats = run_full_eval(df, train_indices, test_indices, train_function, feature_sets, label_column)

    for k, v in predictions.items():
        df.drop(columns=[k], inplace=True, errors='ignore')
        df.loc[test_indices, k] = v

    df.to_pickle('data/all_results_study1.pkl')
    with open('data/coefs.pkl', 'wb') as f:
       pickle.dump(coefs, f)


    # RUN STUDY 2 (mild hypoglycemia) – EVALUATION
    print(f'{"=" * 80}\nRUN STUDY 2\n{"=" * 80}')
    train_function = train_pipe_val  # ATTENTION: SET TO VALIDATION TRAIN
    train_indices, test_indices = df['train'] & (df['subject_id'].isin(STUDY_1_SUBJECTS)), ~df['train'] & df['subject_id'].isin(STUDY_2_SUBJECTS)
    predictions, coefs, stats, raw_stats = run_full_eval(df, train_indices, test_indices, train_function, feature_sets, label_column)

    for k, v in predictions.items():
        df.drop(columns=[k], inplace=True)
        df.loc[test_indices, k] = v
    df.to_pickle('data/all_results_study2.pkl')
    with open('data/coefs2.pkl', 'wb') as f:
       pickle.dump(coefs, f)

    return


if __name__ == '__main__':
    run_evaluation(window_size_sec=WINDOW_SIZE)
