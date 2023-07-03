import os
import numpy as np 
import gc
import shutil
import glob
 
# import cudf

import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags
import pandas as pd
# avoid numba warnings
from numba import config
 
def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin
 
def get_cycled_feature_value_cos(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_cos = np.cos(2*np.pi*value_scaled)
    return value_cos
 
# Relative price to the average price for the category_id
def relative_price_to_avg_categ(col, gdf):
    epsilon = 1e-5
    col = ((gdf['price'] - col) / (col + epsilon)) * (col > 0).astype(int)
    return col
 
class ItemRecency(nvt.ops.Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            item_first_timestamp = gdf['prod_first_event_time_ts']
            delta_days = (col - item_first_timestamp) / (60*60*24)
            gdf[column + "_age_days"] = delta_days * (delta_days >=0)
        return gdf
    @property
    def dependencies(self):
         return ["prod_first_event_time_ts"]
 
    def output_column_names(self, columns):
        return ColumnSelector([column + "_age_days" for column in columns.names])
 
# def api():
#     data = {
#     'event_time': ['2023-01-01 12:00:00', '2023-01-01 12:01:00', '2023-01-02 10:00:00', '2023-01-02 10:02:00', '2023-01-03 15:30:00'],
#     'event_type': ['EventType.view', 'EventType.view', 'EventType.view', 'EventType.view', 'EventType.view'],
#     'product_id': ['A', 'B', 'C', 'D', 'E'],
#     'category_id': [1, 2, 3, 4, 5],
#     'category_code': ['X', 'Y', 'Z', 'X', 'Y'],
#     'brand': ['Brand A', 'Brand B', 'Brand C', 'Brand A', 'Brand B'],
#     'price': [100, 200, 150, 300, 250],
#     'user_id': [587769686, 587769686, 587769686, 587769686, 587769686],
#     'user_session': ['179879', '179879', '179879', '179879', '179879']
#     }


#     # print(data)


#     raw_df = pd.DataFrame(data)
#     # raw_df['event_time'] = pd.to_datetime(raw_df['event_time'])
#     preprocessing(raw_df)
 
 
def preprocessing(raw_df):
    raw_df['event_time_dt'] = raw_df['event_time'].astype('datetime64[s]')
    raw_df['event_time_ts']= raw_df['event_time_dt'].astype('int')
    raw_df = raw_df[raw_df['user_session'].isnull()==False]
    raw_df = raw_df.drop(['event_time'],  axis=1)
    cols = list(raw_df.columns)

    print(cols)
    cols.remove('user_session')
 
    # load data 
    df_event = nvt.Dataset(raw_df) 

    print(df_event)
 
    # categorify user_session 
    cat_feats = ['user_session'] >> nvt.ops.Categorify()
 
    workflow = nvt.Workflow(cols + cat_feats)
    print(workflow)
    workflow.fit(df_event)
    df = workflow.transform(df_event).to_ddf().compute()
    raw_df = None
    del(raw_df)
    gc.collect()
 
    df = df.sort_values(['user_session', 'event_time_ts']).reset_index(drop=True)
 
    print("Count with in-session repeated interactions: {}".format(len(df)))
    # Sorts the dataframe by session and timestamp, to remove consecutive repetitions
    df['product_id_past'] = df['product_id'].shift(1).fillna(0)
    df['session_id_past'] = df['user_session'].shift(1).fillna(0)
    #Keeping only no consecutive repeated in session interactions
    df = df[~((df['user_session'] == df['session_id_past']) & \
              (df['product_id'] == df['product_id_past']))]
    print("Count after removed in-session repeated interactions: {}".format(len(df)))
    del(df['product_id_past'])
    del(df['session_id_past'])
    gc.collect()
 
    item_first_interaction_df = df.groupby('product_id').agg({'event_time_ts': 'min'}) \
    .reset_index().rename(columns={'event_time_ts': 'prod_first_event_time_ts'})
    gc.collect()
 
    df = df.merge(item_first_interaction_df, on=['product_id'], how='left').reset_index(drop=True)
    del(item_first_interaction_df)
    item_first_interaction_df=None
    gc.collect()
 
    df = df[df['event_time_dt'] < np.datetime64('2020-01-06')].reset_index(drop=True)
    df = df.drop(['event_time_dt'],  axis=1)
 
    cat_feats = ['user_session', 'category_code', 'brand', 'user_id', 'product_id', 'category_id', 'event_type'] >> nvt.ops.Categorify(start_index=1)
    # create time features
    session_ts = ['event_time_ts']
 
    session_time = (
        session_ts >> 
        nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='s')) >> 
        nvt.ops.Rename(name = 'event_time_dt')
    )
 
    sessiontime_weekday = (
        session_time >> 
        nvt.ops.LambdaOp(lambda col: col.dt.weekday) >> 
        nvt.ops.Rename(name ='et_dayofweek')
    )
 
 
    weekday_sin = sessiontime_weekday >> (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_sin')
    weekday_cos= sessiontime_weekday >> (lambda col: get_cycled_feature_value_cos(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_cos')
 
    recency_features = ['event_time_ts'] >> ItemRecency() 
    # Apply standardization to this continuous feature
    recency_features_norm = recency_features >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='product_recency_days_log_norm')
 
    time_features = (
        session_time +
        sessiontime_weekday +
        weekday_sin +
        weekday_cos +
        recency_features_norm
    )
 
    price_log = ['price'] >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='price_log_norm')
    avg_category_id_pr = ['category_id'] >> nvt.ops.JoinGroupby(cont_cols =['price'], stats=["mean"]) >> nvt.ops.Rename(name='avg_category_id_price')
    relative_price_to_avg_category = avg_category_id_pr >> nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=['price']) >> nvt.ops.Rename(name="relative_price_to_avg_categ_id")
    groupby_feats = ['event_time_ts', 'user_session'] + cat_feats + time_features + price_log + relative_price_to_avg_category
 
    # Define Groupby Workflow
    groupby_features = groupby_feats >> nvt.ops.Groupby(
        groupby_cols=["user_session"], 
        sort_cols=["event_time_ts"],
        aggs={
            'user_id': ['first'],
            'product_id': ["list", "count"],
            'category_code': ["list"],  
            'event_type': ["list"], 
            'brand': ["list"], 
            'category_id': ["list"], 
            'event_time_ts': ["first"],
            'event_time_dt': ["first"],
            'et_dayofweek_sin': ["list"],
            'et_dayofweek_cos': ["list"],
            'price_log_norm': ["list"],
            'relative_price_to_avg_categ_id': ["list"],
            'product_recency_days_log_norm': ["list"]
        },
        name_sep="-")
 
    groupby_features_list = groupby_features['product_id-list',
                                             'category_code-list',  
                                             'event_type-list', 
                                             'brand-list', 
                                             'category_id-list', 
                                             'et_dayofweek_sin-list',
                                             'et_dayofweek_cos-list',
                                             'price_log_norm-list',
                                             'relative_price_to_avg_categ_id-list',
                                             'product_recency_days_log_norm-list']
 
    SESSIONS_MAX_LENGTH = 20 
    MINIMUM_SESSION_LENGTH = 2
 
    groupby_features_trim = groupby_features_list >> nvt.ops.ListSlice(0,SESSIONS_MAX_LENGTH) >> nvt.ops.Rename(postfix = '_seq')
 
    # calculate session day index based on 'timestamp-first' column
    day_index = ((groupby_features['event_time_dt-first'])  >> 
                 nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
                 nvt.ops.Rename(f = lambda col: "day_index")
                )
 
    selected_features = groupby_features['user_session', 'product_id-count'] + groupby_features_trim + day_index
 
    filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["product_id-count"] >= MINIMUM_SESSION_LENGTH)
 
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
 
    dataset = nvt.Dataset(df)
    print(dataset)
 
    workflow = nvt.Workflow(filtered_sessions)
    workflow.fit(dataset)
    sessions_gdf = workflow.transform(dataset).to_ddf()

    return True

# api()


# raw_df = pd.DataFrame(data)

# api()