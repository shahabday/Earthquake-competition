import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath('../src'))
from utils import *
from data_cleaning import *
from pipeline import *


# load the data
df_X, df_y = load_data_train()


# convert the data to objects
df_X = convert_to_object(df_X)
df_X['geo_level_1_id'] = df_X['geo_level_1_id'].astype('object')
df_X['geo_level_2_id'] = df_X['geo_level_2_id'].astype('object')
df_X['geo_level_3_id'] = df_X['geo_level_3_id'].astype('object')


# drop duplicates
df_X, df_y = drop_duplicates(df_X,df_y)


# remove outliers
outliers_ids = get_outliers_ids(df_X)
df_X, df_y = drop_row(outliers_ids.tolist(), df_X, df_y)


# split data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42, stratify=df_y['damage_grade'])


# resampling (resampling)
#X_train, y_train =  resample_data(X_train, y_train, 'both')


# dropping id column
y_train, y_test = y_train.drop(['building_id'], axis=1), y_test.drop(['building_id'], axis=1)
X_train, X_test = X_train.drop(['building_id'], axis=1), X_test.drop(['building_id'], axis=1)


# getting numeric and categorical features
numerical_feature = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_feature = X_train.select_dtypes(include=['object']).columns.tolist()


# selecting feat
standard_scaler_cols = numerical_feature
robust_scaler_cols = []
baseN_enc_cols = categorical_feature
ordinal_enc_cols = []
one_hot_cols = []


# models
model = RandomForestClassifier(n_estimators=100, random_state=0)


# fit the model
pre_proccessor = pipeline_preprocessor(standard_scaler_cols=standard_scaler_cols,baseN_enc_cols=baseN_enc_cols)
pipeline = classifier_pipeline(pre_proccessor, model)
model_fit = model_training(X_train,y_train,pipeline)


# assess performance of the model
y_pred = model_fit.predict(X_test)
score = f1_score(y_test, y_pred, average='micro')
print(score)
print_confusion_matrix(y_test, y_pred)


# test the model
df_test = load_data_test()
df_test = convert_to_object(df_test)
df_test_id = df_test.pop('building_id')


# get submission
test_predictions = model_fit.predict(df_test)
df_submission = pd.DataFrame({'building_id': df_test_id, 'damage_grade': test_predictions})
df_submission.to_csv('../results/20250130_10_48_submission.csv', index=False)