import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.metrics import RootMeanSquaredError


def _encode_dates(X):
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Encode the date information from the DateOfDeparture columns
    #X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    #X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    #X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])

def create_NN(dropout=0.2, ki='normal'):
    model = Sequential()
    
    model.add(Dense(256,activation='relu',kernel_initializer=ki))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(128,activation='relu',kernel_initializer=ki))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(64,activation='relu',kernel_initializer=ki))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(32,activation='relu',kernel_initializer=ki))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(32,activation='relu',kernel_initializer=ki))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(16,activation='relu',kernel_initializer=ki))
    model.add(BatchNormalization())

    model.add(Dense(1,kernel_initializer=ki))

    model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.01), metrics=[RootMeanSquaredError()])

    return model

def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Parse date to also be of dtype datetime
    data_weather = pd.read_csv(filepath, parse_dates=["Date"])

    X_weather = data_weather[['Date', 'AirPort','CloudCover',
                              'Mean Wind SpeedKm/h']]
    
    #X_weather['Precipitationmm'] = X_weather["Precipitationmm"].replace('T', 0.00, regex=True).astype(float)

    #X_weather["Events"] = X_weather["Events"].replace(np.nan, 'No_info', regex=True)
    #X_weather["No_info"] = [1 if "No_info" in ele else 0 for ele in X_weather["Events"]]
    #X_weather["Thunderstorm"] = [1 if "Thunderstorm" in ele else 0 for ele in X_weather["Events"]]
    #X_weather["Fog"] = [1 if "Fog" in ele else 0 for ele in X_weather["Events"]]
    #X_weather["Snow"] = [1 if "Snow" in ele else 0 for ele in X_weather["Events"]]
    #X_weather["Hail"] = [1 if "Hail" in ele else 0 for ele in X_weather["Events"]]
    #X_weather["Tornado"] = [1 if "Tornado" in ele else 0 for ele in X_weather["Events"]]

    #X_weather = X_weather.drop('Events', axis = 1)
    
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'}
    )

    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )

    X_weather = X_weather.rename(
        columns={'Arrival': 'Departure','CloudCover':'CloudCover_d',
                 'Mean Wind SpeedKm/h':'Mean Wind SpeedKm/h_d'
                 }
    )

    X_merged2 = pd.merge(
        X_merged, X_weather, how='left', on=['DateOfDeparture', 'Departure'], sort=False
    )
    
    return X_merged2

def get_estimator():
    data_merger = FunctionTransformer(_merge_external_data)
    
    date_encoder = FunctionTransformer(_encode_dates)

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = [
        "Arrival", "Departure", "n_days",
        "weekday", "week"
    ]

    numerical_scaler = StandardScaler()
    numerical_cols = ["WeeksToDeparture", "std_wtd",
                      'Mean Wind SpeedKm/h','Mean Wind SpeedKm/h_d',
                      'CloudCover','CloudCover_d'
                      ]

    preprocessor = make_column_transformer(
        (categorical_encoder, categorical_cols),
        (numerical_scaler, numerical_cols)
    )

    params_Lasso = {'alpha': [1,0.1,0.01,0.001,0.0001,0] , "fit_intercept": [True, False]}

    lasso_cv = GridSearchCV(Lasso(),param_grid=params_Lasso, n_jobs=-1, cv=3, refit=True)

    nn = KerasRegressor(build_fn=create_NN,verbose=1,epochs=250, batch_size=64)

    return make_pipeline(data_merger, date_encoder, preprocessor, nn)
