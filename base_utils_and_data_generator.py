import pandas as pd
import numpy as np
import dateutil
from datetime import datetime
from datetime import timedelta
import tensorflow as tf
import sklearn
from sklearn import preprocessing
from sklearn import *
import random
import matplotlib.pyplot as plt
import utils.augmentation as aug
import utils.helper as hlp
import h5py
import gc
from data_visualization import *

if __name__ == '__main__':
    # dir and filename of dataset
    file_DJIA = "dataset_csv/DJIA index_stooq_daily_since18960527.csv"
    file_SP500 = "dataset_csv/Standard and Poor's (S&P) 500 Index Data including Dividend, Earnings and PE Ratio_datahub_monthly_since18710101.csv"
    file_10_year_US_Government_Bond_Yields = "dataset_csv/10 year US Government Bond Yields (long-term interest rate)_datahub_monthly_since19530430to20200731.csv"
    file_10_Year_Treasury_Constant_Maturity_Rate = "dataset_csv/10-Year Treasury Constant Maturity Rate_FRED stlouisfed_daily_since19620102.csv"
    file_CPI_PPI_PCE_US = "dataset_csv/CPI PPI PCE_US_FRED stlouisfed_monthly_since 19500101.csv"
    file_Effective_Federal_Funds_Rate = "dataset_csv/Effective Federal Funds Rate_US_FRED stlouisfed_monthly_since19540701.csv"
    file_GDP_of_US = "dataset_csv/GDP of US_fred stlouisfed_quarterly_since19470401.csv"
    file_Gold_Price = "dataset_csv/Gold Price_investing_monthly_since195001.csv"
    file_Interest_Rates_Discount_Rate_US = "dataset_csv/Interest Rates, Discount Rate for United States_Fred stlouisfed_monthly_since19500101.csv"
    file_NASDAQ_Composite = "dataset_csv/NASDAQ Composite_Yahoo_daily_since19710208.csv"
    file_Russel_2000 = "dataset_csv/Russel 2000_Yahoo_daily_since19870911.csv"
    file_unemployment_rate = "dataset_csv/unemployment rate_us_fred stlouisifed_quaterly_since19480101.csv"
    file_US_dollar_exchange_rates = "dataset_csv/US dollar exchange rates_BIS_Different Countries_daily_since different time.csv"

    # hyper-parameters
    unemployment_rate_binning_list = [0, 4, 5, 6, 7, 8, 100]
    #ratio_matrix = [[30, 0.03]]
    ratio_matrix = [[5, 0.03]]
    range_of_days_before_the_drop_day = [-180, 0]
    length_of_days_before_the_drop_day = range_of_days_before_the_drop_day[1] - range_of_days_before_the_drop_day[0]
    #number_of_days_between_two_dropping_date = 7
    number_of_days_between_two_dropping_date = 0
    # front_margin_no_drop_day, back_margin_no_drop_day = 30, 60
    front_margin_no_drop_day, back_margin_no_drop_day = 0, 0
    ratio_of_label0_to_label1 = 1
    #balanced_testing_dataset = False
    balanced_training_dataset = True
    if_data_augmentation = True
    data_augmentation_method = "mixed"
    # data_augmentation_method can be ["jitter", "scaling", "magnitude_warp", "window_warp", "mixed"]
    ADD_ONE_PADDING = True

def preprocessing_dataframe_to_days(origin_dataframe):
    # 1. Lowercase all column to search for some certain keyword, such as "date"
    origin_dataframe.columns = map(str.lower, origin_dataframe.columns)
    origin_dataframe['date'] = pd.to_datetime(origin_dataframe['date'])
    # 2. get start and end date
    start_date, end_date = origin_dataframe.loc[:, 'date'].iloc[0], origin_dataframe.loc[:, 'date'].iloc[-1]
    # 3. set date as index and sort with index
    origin_dataframe.set_index("date", inplace=True)
    origin_dataframe.sort_index(inplace=True)
    # origin_dataframe.to_csv('temp_concat_df_origin_dataframe_before_shift.csv')
    # origin_dataframe = origin_dataframe.iloc[:, :].shift(1)
    # origin_dataframe.to_csv('temp_concat_df_origin_dataframe_after_shift.csv')
    # 4. create a new df with freq as DAY, start and end with the same time as original dataframe. using datetime as index,
    # use freq = B to create the table with only business day.
    temp_dataframe = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq="B"), columns=["date"])
    temp_dataframe.iloc[:,0] = pd.to_datetime(temp_dataframe.iloc[:,0])
    temp_dataframe.set_index("date", inplace=True)
    # 5. concat the new dataframe with original dataset
    temp_merge_df = pd.concat([temp_dataframe, origin_dataframe], axis=1)
    # 6. fill NaN with recent data
    temp_merge_df = temp_merge_df.fillna(method='ffill')
    # 7. only business day
    temp_merge_df = pd.merge(temp_dataframe, temp_merge_df, how='left', left_index=True, right_index=True,
                           sort=True,
                           copy=True,
                           indicator=False)

    # return start_date, end_date, temp_merge_df
    return temp_merge_df


def there_is_a_sudden_drop(DJIA_dataframe, drop_time_amp_ratio_matrix, number_of_days_between_two_dropping_date):
    # drop_time_amp_ratio_matrix is a matrix describe what situation will be defined as a drop.
    # e.g., the drop_time_amp_ratio_matrix is [[5,10]]
    # means: The date that meet this condition will be marked: compared to the price of the current day,
    #       the mean of the price of the next 5 days fell by 10% or more.
    # e.g., the drop_time_amp_ratio_matrix is [[1,6],[5,10],[45,15]]
    # means: The date that meet this condition will be marked: compared to the price of the current day,
    #       the mean of the price of the next 1, 5, 45 days fell by 6%, 10%, 15% or more respectively.
    # And the day that meet this condition will be defined as a dropping day.
    # e.g. date_set_list = there_is_a_sudden_drop(DJIA.iloc[:500], ratio_matrix)

    # number_of_days_between_two_dropping_date is the days between two dropping days
    # e.g when it is set to 45, that means after the date that is count as a dropping date, the following 45 days will not be count as a dropping date.

    number_of_condition = len(drop_time_amp_ratio_matrix)
    dropping_days_df = pd.DataFrame([])
    previous_dropping_date = datetime(1850, 1, 1, 00, 00)
    length_of_DJIA_df = len(DJIA_dataframe.loc[:, "close"])
    #print("length_of_DJIA_df", length_of_DJIA_df)
    #count_else = 0

    for index, current_day_price in enumerate(DJIA_dataframe.loc[:, "close"]):
        number_of_condition_meets = 0
        for drop_time_amp in drop_time_amp_ratio_matrix:
            drop_period, drop_amp = drop_time_amp[0], drop_time_amp[1]
            # 1. This "if" below is defined by the mean of the index in the specified length of future days.
            # if DJIA_dataframe.loc[:, "close"].iloc[index+1: index+1+drop_period].mean() <= current_day_price*(1-drop_amp):
            # 2. This "if" below is defined by the index on the specified future day.
            #if DJIA_dataframe.loc[:, "close"].iloc[index + drop_period] <= current_day_price * (1 - drop_amp):
            # 3. This "for and if" below is defined by the rule that is any date within the drop period in the future is less than the drop_amp, and the date will be counted as a drop date.
            for date_in_future_within_drop_period in range(1, drop_period+1):
                if index + date_in_future_within_drop_period<length_of_DJIA_df and DJIA_dataframe.loc[:, "close"].iloc[index + date_in_future_within_drop_period] <= current_day_price*(1 - drop_amp):
                    # print("index", index)
                    # print("current_day_price", current_day_price)
                    # print('DJIA_dataframe.loc[:, close].iloc[index + date_in_future_within_drop_period]', DJIA_dataframe.loc[:, "close"].iloc[index + date_in_future_within_drop_period])
                    # print("current_day_price*(1 - drop_amp)", current_day_price*(1 - drop_amp))
                    number_of_condition_meets += 1
                    break
        if number_of_condition_meets == number_of_condition:
            # calculate the days from this dropping date to last dropping date
            temp_number_of_days_between_two_dropping_date = DJIA_dataframe.iloc[[index]].index - previous_dropping_date
            # change the type from timedelta64 to int
            temp_number_of_days_between_two_dropping_date = temp_number_of_days_between_two_dropping_date.astype('timedelta64[D]').astype(int)
            if temp_number_of_days_between_two_dropping_date > number_of_days_between_two_dropping_date:
                dropping_days_df = dropping_days_df.append\
                    (pd.DataFrame(DJIA_dataframe.iloc[[index]].index), ignore_index=True)
            # else:
            #     count_else+=1
                #print("count_else", count_else)
            previous_dropping_date = DJIA_dataframe.iloc[[index]].index
    #print(dropping_days_df)
    return dropping_days_df

def there_is_no_drop_around(DJIA_dataframe, dropping_days_list, front_margin, back_margin, num_of_days):
    # Find the days that have no sudden drop around (within the range of "front_margin" and "back_margin")
    # e.g., When front_margin = 30, back_margin = 90, means there is no sudden drop for at least 30 days before this day,
    # and no sudden drop for at least 90 days after this day.
    # "num_of_days" is to determine how many days will be returned.
    # e.g. date_normal_set_list = there_is_no_drop_around(DJIA.iloc[:500], date_set_list, 10, 20, 6)
    notwarned = True
    max_num_run = num_of_days * 10
    max_number_DJIA_rows = len(DJIA_dataframe.index)
    normal_days_df = pd.DataFrame([])

    # print(dropping_days_list)
    # print("_____________________")
    while max_num_run:
        pick_a_date = DJIA_dataframe.index[np.random.randint(front_margin, max_number_DJIA_rows - 1 - back_margin)]
        try:
            dropping_date_before_pick_a_date = dropping_days_list[dropping_days_list["date"] <= pick_a_date].loc[:, "date"].iloc[-1]
            dropping_date_after_pick_a_date = dropping_days_list[dropping_days_list["date"] > pick_a_date].loc[:, "date"].iloc[0]
            front_margin_from_pick_a_date = pick_a_date - dateutil.relativedelta.relativedelta(days=front_margin)
            back_margin_from_pick_a_date = pick_a_date + dateutil.relativedelta.relativedelta(days=back_margin)
            if (front_margin_from_pick_a_date > dropping_date_before_pick_a_date) and (back_margin_from_pick_a_date < dropping_date_after_pick_a_date):
                normal_days_df = normal_days_df.append({'date': pick_a_date}, ignore_index=True)
                normal_days_df.drop_duplicates(subset=["date"], keep='first', inplace=True)
                if len(normal_days_df.index) == num_of_days:
                    break
            max_num_run -= 1
            if not max_num_run:
                print("[there_is_no_drop_around function]: Warning: Dataset with number less than required is generated. ")
        except:
            if notwarned:
                notwarned = False
                print("[there_is_no_drop_around function]: Warning: Somethings wrong for getting the date before or after dropping date. "
                      "If the result is good, this can be ignored.")
            max_num_run -= 1
            if not max_num_run:
                print("[there_is_no_drop_around function]: Warning: Dataset with number less than required is generated. ")
            pass
    return normal_days_df

def retrieve_features_set_from_day(features_set, the_day, range_of_days_before_the_day, adding_one_padding=False):
    # range_of_days_before_the_day is a set to describe the range of the period before the dropping date
    # e.g. [-90, -20] means we retrieve the data from 90 days before the dropping date to 20 days before it.
    # e.g. [-60, 0] means we retrieve the data from 60 days before dropping date to that date.
    # e.g. features_subset = retrieve_features_set_from_day(SP500, date_set_list.loc[1, "date"], 10)



    if features_set.index[-range_of_days_before_the_day[0]] > the_day:
        # print("features_set.index[-range_of_days_before_the_day[0]]", features_set.index[-range_of_days_before_the_day[0]])
        # print("the_day ", the_day)
        # print("retrieve_features_set_from_day: False")
        return False
    else:
        if range_of_days_before_the_day[1] == 0:
            temp_features_set = features_set[features_set.index < the_day].iloc[range_of_days_before_the_day[0]:]
        else:
            temp_features_set = features_set[features_set.index < the_day].iloc[range_of_days_before_the_day[0]:range_of_days_before_the_day[1]]
        # print("retrieve_features_set_from_day: temp_features_set", temp_features_set.shape)
        # print(type(temp_features_set))
        # print(temp_features_set.iloc[-2:, :])
        # print("retrieve_features_set_from_day: temp_features_set", temp_features_set)

        #adding_one_padding = True
        # print("adding_one_padding", adding_one_padding)
        if adding_one_padding:
            temp_features_set.loc['padding'] = 1
        # print("retrieve_features_set_from_day: temp_features_set", temp_features_set.shape)
        # print(type(temp_features_set))
        # print(temp_features_set.iloc[-2:, :])
        return temp_features_set

def generate_dataset(features_set, the_day_set, label_is_drop, range_of_days_before_the_day, adding_one_padding=False):
    print("generate_dataset adding_one_padding", adding_one_padding)
    # range_of_days_before_the_day is a set to describe the range of the period before the dropping date
    # e.g. [-90, -20] means we retrieve the data from 90 days before the dropping date to 20 days before it.
    # e.g. [-60, 0] means we retrieve the data from 60 days before dropping date to that date.
    range_of_days = range_of_days_before_the_day[1] - range_of_days_before_the_day[0]
    # d1: index of date of dropping days, d2: index of days before drop day, d3: features of each days in d2.
    if adding_one_padding:
        X_set = np.empty([the_day_set.shape[0], range_of_days+1, features_set.shape[1]], dtype=float)
    else:
        X_set = np.empty([the_day_set.shape[0], range_of_days, features_set.shape[1]], dtype=float)
    # X_set = np.empty([the_day_set.shape[0]], dtype=object)
    Y_set = np.empty([the_day_set.shape[0], 2], dtype=int)
    # training_set have 3 dimensions, (1: the index of data, (2: date, 3: featrues) as dataframe) as array.
    for index, the_day in the_day_set.iterrows():
        # print("retrieve_features_set_from_day(features_set, the_day['date'], range_of_days_before_the_day)", retrieve_features_set_from_day(features_set, the_day["date"], range_of_days_before_the_day))
        X_set[index] = retrieve_features_set_from_day(features_set, the_day["date"], range_of_days_before_the_day, adding_one_padding=adding_one_padding).values
        # if X_set[index] is False:
        #     raise("Error: retrieve_features_set_from_day, cannot retrieve data in the range.")
        # if the label_is_drop is set to 1, which means it is a positive, then the label is set to 1, the one-hot is set to [0,1];
        # if the label_is_drop is set to 0, which means it is a negative, then the label is set to 0, the one-hot is set to [1,0];
        if label_is_drop:
            Y_set[index] = [0, 1]
        else:
            Y_set[index] = [1, 0]
    # print("X_set", X_set.shape)
    # print("X_set", X_set)
    return X_set, Y_set


def concat_dataset(dataset_df_list, dataset_name_list):
    temp_concat = dataset_df_list[0]
    for index in range(1, len(dataset_df_list)):
        # print("index", index)
        # print("left dataset_name_list[index-1]", dataset_name_list[index-1])
        # print("right dataset_name_list[index]", dataset_name_list[index])
        # print("temp_concat.index[0], temp_concat.index[-1]")
        # print(temp_concat.index[0], temp_concat.index[-1])
        # temp_concat = pd.merge(temp_concat, dataset_df_list[index], how='inner', left_index=True, right_index=True, sort=True,
        #       suffixes=("_"+dataset_name_list[index-1], "_"+dataset_name_list[index]), copy=True, indicator=False)
        temp_concat = pd.merge(temp_concat, dataset_df_list[index], how='inner', left_index=True, right_index=True, sort=True,
              suffixes=("", "_"+dataset_name_list[index]), copy=True, indicator=False)
    # print("temp_concat.index[0], temp_concat.index[-1]")
    # print(temp_concat.index[0], temp_concat.index[-1])
    return temp_concat, temp_concat.index[0], temp_concat.index[-1]


def preprocessing_dataframe_binning(dataset_df, binning_list):
    dataset_df['onehot_labels'] = pd.cut(dataset_df.iloc[:, 0], bins=binning_list, labels=[i for i in range(len(binning_list)-1)], precision=1)
    #dataset_df.to_csv('temp_concat_df_binning_dataset_df_inside.csv')
    temp_df = pd.get_dummies(dataset_df['onehot_labels'], prefix='one_hot')
    return temp_df


def preprocessing_dataframe_minmaxscaler(dataset_df, scalar):
    if not scalar:
        scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
        temp_dataset_df = pd.DataFrame(scalar.fit_transform(dataset_df), index=dataset_df.index,
                                       columns=dataset_df.columns)
    else:
        temp_dataset_df = pd.DataFrame(scalar.transform(dataset_df), index=dataset_df.index,
                                       columns=dataset_df.columns)
    return scalar, temp_dataset_df


def pre_data_augmentation(dataset_array_batch, augmentation_size_times, method="jitter"):
    temp_dataset_array_batch = np.empty(
        [augmentation_size_times * dataset_array_batch.shape[0], dataset_array_batch.shape[1],
         dataset_array_batch.shape[2]])
    method_value = method
    method_list = ["jitter", "scaling", "magnitude_warp", "window_warp"]
    for i in range(augmentation_size_times):
        # for index, each_batch in enumerate(dataset_array_batch):
        if method == "mixed":
            method_value = random.choice(method_list)
            #print("method_value", method_value)

        if method_value == "jitter":
            #print("1 jitter")
            temp_dataset_array_batch[
            i * dataset_array_batch.shape[0]:(i + 1) * dataset_array_batch.shape[0]] = aug.jitter(dataset_array_batch,
                                                                                                  sigma=0.01)
        elif method_value == "scaling":
            #print("2 scaling")
            temp_dataset_array_batch[
            i * dataset_array_batch.shape[0]:(i + 1) * dataset_array_batch.shape[0]] = aug.scaling(dataset_array_batch,
                                                                                                   sigma=0.1)
        elif method_value == "magnitude_warp":
            #print("3 magnitude_warp")
            temp_dataset_array_batch[
            i * dataset_array_batch.shape[0]:(i + 1) * dataset_array_batch.shape[0]] = aug.magnitude_warp(
                dataset_array_batch, sigma=0.2, knot=4)
        elif method_value == "window_warp":
            #print("4 window_warp")
            temp_dataset_array_batch[
            i * dataset_array_batch.shape[0]:(i + 1) * dataset_array_batch.shape[0]] = aug.window_warp(
                dataset_array_batch, window_ratio=0.1, scales=[0.5, 2.])
        else:
            return False
    return temp_dataset_array_batch


def balance_dataset(datasetX_label0, Y_label0, datasetX_label1, Y_label1):
    length = np.min([datasetX_label0.shape[0], datasetX_label1.shape[0]])
    np.random.shuffle(datasetX_label0)
    np.random.shuffle(datasetX_label1)
    return datasetX_label0[:length], Y_label0[:length], datasetX_label1[:length], Y_label1[:length]


# data generator
def time_series_data_generator_with_window(features_set, window_size, start_seq, ret_length, adding_one_padding=False, return_next_day_price=False):
    # feature_set is the dataset with features
    # window_size is size of the moving window, e.g. [-90, -20], [-60, 0]
    # start_date is the seq that start the scan, e,g "31/12/1999"
    # ret_length is the length of returned array

    window_size_len = window_size[1]-window_size[0]
    num_of_dim = features_set.shape[1]
    ret_date = np.empty([ret_length], dtype=datetime)
    ret_current_day_close_price = np.empty([ret_length])
    # if we want the next day price for regression
    if return_next_day_price: ret_next_day_price = np.empty([ret_length])
    # if we want to add a padding after the last day to be the self-attention output position
    if adding_one_padding:
        ret_feature = np.empty([ret_length, window_size_len+1, num_of_dim])
    else:
        ret_feature = np.empty([ret_length, window_size_len, num_of_dim])

    for index in range(ret_length):
        if return_next_day_price:
            last_index = start_seq + index + 1
        else:
            last_index = start_seq + index
        if last_index >= features_set.shape[0]:
            #print("Out of range: start_seq+index: ", start_seq+index, "index: ", index, "features_set.shape[0]: ", features_set.shape[0])
            ret_feature = ret_feature[:index]
            ret_date = ret_date[:index]
            ret_current_day_close_price = ret_current_day_close_price[:index]
            if return_next_day_price:
                ret_next_day_price = ret_next_day_price[:index]
            break
        cur_date = pd.Timestamp(features_set.iloc[[start_seq+index]].index.to_pydatetime()[0])
        ret_date[index] = features_set.iloc[[start_seq+index]].index.to_pydatetime()[0]
        ret_current_day_close_price[index] = features_set.iloc[[start_seq + index]]["close"]
        ret_feature[index] = retrieve_features_set_from_day(features_set, cur_date, window_size, adding_one_padding=adding_one_padding)
        if return_next_day_price:
            ret_next_day_price[index] = features_set.iloc[[start_seq + index+1]]["close"]
        # cur_date = cur_date + timedelta(days=1)
    # print("ret_length", ret_length)
    # print("ret_feature.shape", ret_feature.shape)
    # print("ret_feature[:3]", ret_feature[:3])
    # print("ret_feature[-3:]", ret_feature[-3:])
    # print("ret_date.shape", ret_date.shape)
    # print("ret_date[:3]", ret_date[:3])
    # print("ret_date[-3:]", ret_date[-3:])
    # print("ret_close_price.shape", ret_current_day_close_price.shape)
    # print("ret_close_price[:3]", ret_current_day_close_price[:3])
    # print("ret_close_price[-3:]", ret_current_day_close_price[-3:])
    # print("ret_next_day_price.shape", ret_next_day_price.shape)
    # print("ret_next_day_price[:3]", ret_next_day_price[:3])
    # print("ret_next_day_price[-3:]", ret_next_day_price[-3:])
    if return_next_day_price:
        return ret_feature, ret_date, ret_current_day_close_price, ret_next_day_price
    return ret_feature, ret_date, ret_current_day_close_price


def separate_training_dataset_and_testing_dataset(features_set, ratio_of_training_dataset):
    print("features_set.shape", features_set.shape)
    length_all = features_set.shape[0]
    length_train = int(length_all*ratio_of_training_dataset)
    training_database = features_set[:length_train]
    testing_database = features_set[length_train:]
    print("training_database.shape", training_database.shape)
    print("testing_database.shape", testing_database.shape)
    return training_database, testing_database


def generate_dataset_for_testing_dataset(testing_database_set, drop_date_set_list, testing_dataset):
    temp_df = pd.DataFrame(columns=[], index=testing_database_set.index)
    temp_df['label_#0'] = 1
    temp_df['label_#1'] = 0

    drop_date_set_list.set_index("date", inplace=True)
    drop_date_set_list['label_#0'] = 0
    drop_date_set_list['label_#1'] = 1
    temp_df.update(drop_date_set_list)

    testing_X = np.array(testing_dataset)
    testing_Y = np.array(temp_df)
    return testing_X, testing_Y


def from_one_hot_to_1d_array(one_hot_label):
    ret = np.empty([one_hot_label.shape[0]])
    for i, label in enumerate(one_hot_label):
        ret[i] = np.argmax(label)
    return ret


if __name__ == '__main__':
    # read to dataframe
    df_DJIA = pd.read_csv(file_DJIA)
    df_SP500 = pd.read_csv(file_SP500)
    df_10_year_US_Government_Bond_Yields = pd.read_csv(file_10_year_US_Government_Bond_Yields)
    df_10_Year_Treasury_Constant_Maturity_Rate = pd.read_csv(file_10_Year_Treasury_Constant_Maturity_Rate)
    df_CPI_PPI_PCE_US = pd.read_csv(file_CPI_PPI_PCE_US)
    df_Effective_Federal_Funds_Rate = pd.read_csv(file_Effective_Federal_Funds_Rate)
    df_file_GDP_of_US = pd.read_csv(file_GDP_of_US)
    df_file_Gold_Price = pd.read_csv(file_Gold_Price)
    df_file_Interest_Rates_Discount_Rate_US = pd.read_csv(file_Interest_Rates_Discount_Rate_US)
    df_file_NASDAQ_Composite = pd.read_csv(file_NASDAQ_Composite)
    df_file_Russel_2000 = pd.read_csv(file_Russel_2000)
    df_unemployment_rate = pd.read_csv(file_unemployment_rate)
    # df_file_US_dollar_exchange_rates = pd.read_csv(file_US_dollar_exchange_rates)

    # adding month column
    df_DJIA['Date'] = pd.to_datetime(df_DJIA['Date'])
    #df_DJIA.to_csv("DJIAbefore.csv")
    df_DJIA['month'] = df_DJIA["Date"].dt.month
    #df_DJIA.to_csv("DJIAafter.csv")

    # preprocessing convert Periodicity to days
    DJIA = preprocessing_dataframe_to_days(df_DJIA)
    SP500 = preprocessing_dataframe_to_days(df_SP500)
    ten_year_US_Government_Bond_Yields = preprocessing_dataframe_to_days(df_10_year_US_Government_Bond_Yields)
    ten_Year_Treasury_Constant_Maturity_Rate = preprocessing_dataframe_to_days(df_10_Year_Treasury_Constant_Maturity_Rate)
    CPI_PPI_PCE_US = preprocessing_dataframe_to_days(df_CPI_PPI_PCE_US)
    Effective_Federal_Funds_Rate = preprocessing_dataframe_to_days(df_Effective_Federal_Funds_Rate)
    GDP_of_US = preprocessing_dataframe_to_days(df_file_GDP_of_US)
    Gold_Price = preprocessing_dataframe_to_days(df_file_Gold_Price)
    Interest_Rates_Discount_Rate_US = preprocessing_dataframe_to_days(df_file_Interest_Rates_Discount_Rate_US)
    NASDAQ_Composite = preprocessing_dataframe_to_days(df_file_NASDAQ_Composite)
    Russel_2000 = preprocessing_dataframe_to_days(df_file_Russel_2000)
    unemployment_rate = preprocessing_dataframe_to_days(df_unemployment_rate)
    # US_dollar_exchange_rates = preprocessing_dataframe_to_days(df_file_US_dollar_exchange_rates)

    # binning
    unemployment_rate_binning = preprocessing_dataframe_binning(unemployment_rate, unemployment_rate_binning_list)
    #unemployment_rate.to_csv('temp_concat_df_before_binning.csv')
    #unemployment_rate_binning.to_csv('temp_concat_df_binning.csv')

    # normalization
    model_scalar_ten_year_US_Government_Bond_Yields, df_ten_year_US_Government_Bond_Yields = preprocessing_dataframe_minmaxscaler(ten_year_US_Government_Bond_Yields, [])
    df_ten_year_US_Government_Bond_Yields.to_csv('temp_concat_df_nor_ten_year_US_Government_Bond_Yields.csv')

    # concat
    # ten_year_US_Government_Bond_Yields.iloc[-700:].to_csv('temp_concat_df_ten_year_US_Government_Bond_Yields_700.csv')
    # ten_Year_Treasury_Constant_Maturity_Rate.iloc[-700:].to_csv('temp_concat_df_ten_Year_Treasury_Constant_Maturity_Rate_700.csv')
    # dataset_df_list = [ten_year_US_Government_Bond_Yields.iloc[-700:], ten_Year_Treasury_Constant_Maturity_Rate.iloc[-700:], CPI_PPI_PCE_US.iloc[-700:], Effective_Federal_Funds_Rate.iloc[-700:]]
    dataset_df_list = [DJIA, SP500, ten_year_US_Government_Bond_Yields, ten_Year_Treasury_Constant_Maturity_Rate, CPI_PPI_PCE_US,
                       Effective_Federal_Funds_Rate, GDP_of_US, Gold_Price, Interest_Rates_Discount_Rate_US, NASDAQ_Composite,
                       Russel_2000, unemployment_rate_binning]
    dataset_name_list = ["DJIA", "SP500", "ten_year_US_Government_Bond_Yields", "ten_Year_Treasury_Constant_Maturity_Rate", "CPI_PPI_PCE_US",
                         "Effective_Federal_Funds_Rate", "GDP_of_US", "Gold_Price", "Interest_Rates_Discount_Rate_US", "NASDAQ_Composite",
                         "Russel_2000", "unemployment_rate_binning"]
    temp_concat, start_datetime, end_datetime = concat_dataset(dataset_df_list, dataset_name_list)
    #temp_concat.to_csv('temp_concat_df.csv')
    # print("temp_concat", temp_concat)

    # training_data_1095, _, _, _ = time_series_data_generator_with_window(temp_concat, [-10, 0], 10, 20,
    #                                                                      adding_one_padding=False,
    #                                                                      return_next_day_price=True)

    # _scalar, temp_concat = preprocessing_dataframe_minmaxscaler(temp_concat, [])
    # temp_concat.to_csv('temp_concat_df_norm.csv')
    # temp_concat = pd.read_csv('temp_concat_df_norm.csv')

    # generate training dataset for autoencoder
    training_database_set, testing_database_set = separate_training_dataset_and_testing_dataset(temp_concat, 0.67)
    minmax_scalar_training_dataset, training_database_set = preprocessing_dataframe_minmaxscaler(training_database_set, [])
    training_data_1095, _, _ = time_series_data_generator_with_window(training_database_set, [-365*3, 0], 365*3, 6870, adding_one_padding=False, return_next_day_price=False)
    print("training_data_1095", training_data_1095.shape)
    print("training_data_1095", training_data_1095[:5])
    # with h5py.File('ae_data_addingDJIA.h5', 'a') as hf:
    #     hf.create_dataset("training_data_1095_addingDJIA_training",  data=training_data_1095)

    #print("training_data_1095", training_data_1095.shape)

    # print(type(temp_concat))
    # training_database_set, testing_database_set = separate_training_dataset_and_testing_dataset(temp_concat, 0.67)
    # print("training_database_set.shape", training_database_set.shape)
    # print("testing_database_set.shape", testing_database_set.shape)
    # window_size = 365*3
    # start_seq = len(training_database_set)
    # testing_dataset, _, _ = time_series_data_generator_with_window(temp_concat, [-window_size, 0], start_seq+window_size, 6870)
    # print("testing_dataset.shape", testing_dataset.shape)


    # generating regression pre-training data
    #training_database_set, testing_database_set = separate_training_dataset_and_testing_dataset(temp_concat, 0.8)
    # _, temp_concat_norm = preprocessing_dataframe_minmaxscaler(temp_concat, [])
    # training_database_set = temp_concat_norm
    # print("training_database_set", training_database_set.shape)
    # print("training_database_set", training_database_set[:2])
    #
    # # getting regression dataset with window of 300
    # training_data_regression_100_X, _, training_data_regression_100_Y = time_series_data_generator_with_window\
    #     (training_database_set, window_size=[-300, 0], start_seq=300, ret_length=10000,
    #      adding_one_padding=False, return_next_day_price=False)
    # print("training_data_regression_100_X", training_data_regression_100_X.shape)
    # print(training_data_regression_100_X[:2])
    # print("training_data_regression_100_Y", training_data_regression_100_Y.shape)
    # print(training_data_regression_100_Y[:2])
    # with h5py.File('training_data_regression_300.h5', 'a') as hf:
    #      hf.create_dataset("training_data_regression_300_X",  data=training_data_regression_100_X)
    #      hf.create_dataset("training_data_regression_300_Y", data=training_data_regression_100_Y)
    # del training_data_regression_100_X, training_data_regression_100_Y
    # gc.collect()
    #
    # # getting regression dataset with window of 500
    # training_data_regression_500_X, _, training_data_regression_500_Y = time_series_data_generator_with_window\
    #     (training_database_set, window_size=[-500, 0], start_seq=500, ret_length=10000,
    #      adding_one_padding=False, return_next_day_price=False)
    # print("training_data_regression_500_X", training_data_regression_500_X.shape)
    # print(training_data_regression_500_X[:2])
    # print("training_data_regression_500_Y", training_data_regression_500_Y.shape)
    # print(training_data_regression_500_Y[:2])
    # with h5py.File('training_data_regression.h5', 'a') as hf:
    #      hf.create_dataset("training_data_regression_500_X",  data=training_data_regression_500_X)
    #      hf.create_dataset("training_data_regression_500_Y", data=training_data_regression_500_Y)
    # del training_data_regression_500_X, training_data_regression_500_Y
    # gc.collect()
    #
    # # getting regression dataset with window of 1000
    # training_data_regression_1000_X, _, training_data_regression_1000_Y = time_series_data_generator_with_window\
    #     (training_database_set, window_size=[-1000, 0], start_seq=1000, ret_length=10000,
    #      adding_one_padding=False, return_next_day_price=False)
    # print("training_data_regression_1000_X", training_data_regression_1000_X.shape)
    # print(training_data_regression_1000_X[:2])
    # print("training_data_regression_1000_Y", training_data_regression_1000_Y.shape)
    # print(training_data_regression_1000_Y[:2])
    # with h5py.File('training_data_regression.h5', 'a') as hf:
    #      hf.create_dataset("training_data_regression_1000_X",  data=training_data_regression_1000_X)
    #      hf.create_dataset("training_data_regression_1000_Y", data=training_data_regression_1000_Y)
    # del training_data_regression_1000_X, training_data_regression_1000_Y
    # gc.collect()