import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

binary_cols = ['Gender', 'Customer Type', 'Type of Travel']
ordinal_cols = ['Class']
rating_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 
    'Ease of Online booking', 'Gate location', 'Food and drink', 
    'Online boarding', 'Seat comfort', 'Inflight entertainment', 
    'On-board service', 'Leg room service', 'Baggage handling', 
    'Checkin service', 'Cleanliness'
]
numeric_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

def encode_categorical(df_train, df_val, df_test=None):
    train, val = df_train.copy(), df_val.copy()
    test = df_test.copy() if df_test is not None else None

    # encode binary categorical columns
    binary_mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'Customer Type': {'Loyal Customer': 1, 'disloyal Customer': 0},
        'Type of Travel': {'Business travel': 1, 'Personal Travel': 0}
    }

    for col, mapping in binary_mapping.items():
        if col in train.columns:
            train[col] = train[col].map(mapping)
            val[col] = val[col].map(mapping)
            if test is not None:
                test[col] = test[col].map(mapping)
    
    # encode ordinal categorical columns
    class_order = [['Eco', 'Eco Plus', 'Business']]
    order_encoder = OrdinalEncoder(categories=class_order, handle_unknown='use_encoded_value', unknown_value=-1)

    train['Class'] = order_encoder.fit_transform(train[['Class']])
    val['Class'] = order_encoder.transform(val[['Class']])
    if test is not None:
        test['Class'] = order_encoder.transform(test[['Class']])

    return train, val, test

def impute_service_ratings(df_train, df_val, df_test=None):
    train, val = df_train.copy(), df_val.copy()
    test = df_test.copy() if df_test is not None else None

    # impute NaNs in ratings (converted from 0s) using median strategy
    imputer = SimpleImputer(strategy='median')
    train[rating_cols] = imputer.fit_transform(train[rating_cols])
    val[rating_cols] = imputer.transform(val[rating_cols])
    if test is not None:
        test[rating_cols] = imputer.transform(test[rating_cols])

    
    return train, val, test

def create_engineered_features(df):
    df = df.copy()
    
    # create delay features
    df['Total_Delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['Delay_Difference'] = df['Arrival Delay in Minutes'] - df['Departure Delay in Minutes']
    df['Has_Delay'] = (df['Total_Delay'] > 0).astype(int)
    
    # aggregate service scores
    df['Mean_Rating'] = df[rating_cols].mean(axis=1)
    df['Max_Rating'] = df[rating_cols].max(axis=1)
    df['Min_Rating'] = df[rating_cols].min(axis=1)
    
    
    return df

def scale_continuous_features(df_train, df_val, cols_to_scale, df_test=None):
    train, val = df_train.copy(), df_val.copy()
    test = df_test.copy() if df_test is not None else None

    scaler = StandardScaler()
    train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
    val[cols_to_scale] = scaler.transform(val[cols_to_scale])
    if test is not None:
        test[cols_to_scale] = scaler.transform(test[cols_to_scale])

    return train, val, test, scaler

def run_full_preprocessing(df_train, df_val, df_test=None):
    # 1. Categorical encoding
    X_tr, X_v, X_te = encode_categorical(df_train, df_val, df_test)

    # 2. Rating imputation
    X_tr, X_v, X_te = impute_service_ratings(X_tr, X_v, X_te)

    # 3. Feature engineering
    X_tr = create_engineered_features(X_tr)
    X_v = create_engineered_features(X_v)
    X_te = create_engineered_features(X_te) if X_te is not None else None
    
    # 4. Continuous feature scaling
    continuous_cols = numeric_cols + ['Total_Delay', 'Delay_Difference', 'Mean_Rating']
    X_tr, X_v, X_te, scaler = scale_continuous_features(X_tr, X_v, continuous_cols, X_te)
    
    return X_tr, X_v, X_te, scaler
