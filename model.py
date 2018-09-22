import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from SchoolBudgets.multilabel_train_test_split import multilabel_train_test_split
from SchoolBudgets.SparseInteractions import SparseInteractions
from sklearn.preprocessing import FunctionTransformer, Imputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

df=pd.read_csv('SchoolBudgets/TrainingData.csv')

# Create the DataFrame: numeric_data_only
NUMERIC_COLUMNS=['FTE', 'Total']
LABELS=['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type',
        'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2,
                                                               seed=123)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))

#Compare accuracy of 0 to end.


# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit it to the training data
clf.fit(X_train, y_train)

# Load the holdout data: holdout
holdout = pd.read_csv('HoldoutData.csv', index_col=0)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)

# Save prediction_df to csv
prediction_df.to_csv("predictions.csv")



## Import CountVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#
## Create the token pattern: TOKENS_ALPHANUMERIC
#TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
#
## Fill missing values in df.Position_Extra
#df.Position_Extra.fillna('',inplace=True)
#
## Instantiate the CountVectorizer: vec_alphanumeric
#vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
#
## Fit to the data
#vec_alphanumeric.fit(df.Position_Extra)
#
## Print the number of tokens and first 15 tokens
#msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
#print(msg.format(len(vec_alphanumeric.get_feature_names())))
#print(vec_alphanumeric.get_feature_names()[:15])



# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """

    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)

    # Replace nans with blanks
    text_data.fillna("",inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)



## Import the CountVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#
## Create the basic token pattern
#TOKENS_BASIC = '\\S+(?=\\s+)'
#
## Create the alphanumeric token pattern
#TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
#
## Instantiate basic CountVectorizer: vec_basic
#vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)
#
## Instantiate alphanumeric CountVectorizer: vec_alphanumeric
#vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
#
## Create the text vector
#text_vector = combine_text_columns(df)
#
## Fit and transform vec_basic
#vec_basic.fit_transform(text_vector)
#
## Print number of tokens of vec_basic
#print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))
#
## Fit and transform vec_alphanumeric
#vec_alphanumeric.fit_transform(text_vector)
#
## Print number of tokens of vec_alphanumeric
#print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))

NON_LABELS=['Object_Description', 'Text_2', 'SubFund_Description', 'Job_Title_Description', 'Text_3', 'Text_4', 'Sub_Object_Description', 'Location_Description', 'FTE', 'Function_Description', 'Facility_or_Department', 'Position_Extra', 'Total', 'Program_Description', 'Fund_Description', 'Text_1']

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2,
                                                               seed=123)

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)



# Complete the pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imp', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(ngram_range=(1,2)))
                    ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset with independent classifier: ", accuracy)


# Edit model step in pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier(n_estimators=15))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset with Random Forest Classifier: ", accuracy)


#Final Pipeline
# Import pipeline
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import other preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest

# Select 300 best features
chi_k = 300

# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Import the hashing vectorizer

p2 = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', XGBClassifier())))
    ])
