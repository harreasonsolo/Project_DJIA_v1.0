from selfattention_model import *



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
    range_of_days_before_the_drop_day = [-400, 0]
    #range_of_days_before_the_drop_day = [-180, 0]
    length_of_days_before_the_drop_day = range_of_days_before_the_drop_day[1] - range_of_days_before_the_drop_day[0]
    #number_of_days_between_two_dropping_date = 30
    number_of_days_between_two_dropping_date = 0
    # front_margin_no_drop_day, back_margin_no_drop_day = 30, 60
    front_margin_no_drop_day, back_margin_no_drop_day = 0, 0
    ratio_of_label0_to_label1 = 2
    # balanced_testing_dataset = False
    balanced_training_dataset = True
    if_data_augmentation = True
    data_augmentation_method = "mixed"
    # data_augmentation_method can be ["jitter", "scaling", "magnitude_warp", "window_warp", "mixed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = 3
    valid_ratio = 0.15
    learning_rate = 1e-3
    epoch_num = 100
    # different train method and target method
    TRAIN_METHOD = "classification" # classification, regression
    TARGET_METHOD = "last_one" # last_one, average, padding_one
    # use padding as last one to do the classification
    ADD_ONE_PADDING = True if TARGET_METHOD == "padding_one" else False

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
    df_DJIA.to_csv("DJIAbefore.csv")
    df_DJIA['month'] = df_DJIA["Date"].dt.month
    df_DJIA.to_csv("DJIAafter.csv")

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
    # unemployment_rate.to_csv('temp_concat_df_before_binning.csv')
    # unemployment_rate_binning.to_csv('temp_concat_df_binning.csv')

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
    # dataset_df_list = [SP500, ten_year_US_Government_Bond_Yields, ten_Year_Treasury_Constant_Maturity_Rate, CPI_PPI_PCE_US,
    #                    Effective_Federal_Funds_Rate, GDP_of_US, Gold_Price, Interest_Rates_Discount_Rate_US, NASDAQ_Composite,
    #                    Russel_2000, unemployment_rate_binning]
    # dataset_name_list = ["SP500", "ten_year_US_Government_Bond_Yields", "ten_Year_Treasury_Constant_Maturity_Rate", "CPI_PPI_PCE_US",
    #                      "Effective_Federal_Funds_Rate", "GDP_of_US", "Gold_Price", "Interest_Rates_Discount_Rate_US", "NASDAQ_Composite",
    #                      "Russel_2000", "unemployment_rate_binning"]
    temp_concat, start_datetime, end_datetime = concat_dataset(dataset_df_list, dataset_name_list)
    # DJIA.to_csv('DJIA.csv')
    # print("temp_concat", temp_concat.shape)
    # temp_concat.to_csv('temp_concat_df.csv')

    # Start to separate the training data and testing data
    training_database_set, testing_database_set = separate_training_dataset_and_testing_dataset(temp_concat, 0.67)

    del temp_concat, DJIA, SP500, ten_year_US_Government_Bond_Yields, ten_Year_Treasury_Constant_Maturity_Rate, CPI_PPI_PCE_US, Effective_Federal_Funds_Rate, GDP_of_US, Gold_Price, Interest_Rates_Discount_Rate_US, NASDAQ_Composite, Russel_2000, unemployment_rate_binning
    gc.collect()

    # 1. generating testing dataset
    # 1.1 normalization using scalar fit with training dataset
    minmax_scalar_training_dataset, _ = preprocessing_dataframe_minmaxscaler(training_database_set, [])
    _, testing_database_set_norm = preprocessing_dataframe_minmaxscaler(testing_database_set, minmax_scalar_training_dataset)
    testing_database_set = testing_database_set_norm

    # 1.2 generating testing dataset from testing database
    window_size = length_of_days_before_the_drop_day
    start_seq = len(training_database_set)
    testing_dataset, _, _ = time_series_data_generator_with_window(testing_database_set, [-window_size, 0], window_size, 10000, adding_one_padding=ADD_ONE_PADDING)
    drop_date_set_list_testing_dataset = there_is_a_sudden_drop(DJIA_dataframe=testing_database_set[length_of_days_before_the_drop_day:],
                                                drop_time_amp_ratio_matrix=ratio_matrix,
                                                number_of_days_between_two_dropping_date=number_of_days_between_two_dropping_date)
    drop_date_set_list_testing_dataset.to_csv('drop_date_set_list_testing_dataset.csv')

    testX, testY = generate_dataset_for_testing_dataset(testing_database_set[length_of_days_before_the_drop_day:], drop_date_set_list_testing_dataset, testing_dataset)
    # print("testX.shape", testX.shape)
    # print("testY.shape", testY.shape)
    del testing_database_set_norm, testing_dataset
    gc.collect()
    # Done.

    # 2. generating training data
    drop_date_set_list_training_dataset = there_is_a_sudden_drop(DJIA_dataframe=training_database_set[length_of_days_before_the_drop_day:],
                                                drop_time_amp_ratio_matrix=ratio_matrix,
                                                number_of_days_between_two_dropping_date=number_of_days_between_two_dropping_date)
    date_normal_set_list_training_dataset = there_is_no_drop_around(DJIA_dataframe=training_database_set[length_of_days_before_the_drop_day:],
                                                   dropping_days_list=drop_date_set_list_training_dataset,
                                                   front_margin=front_margin_no_drop_day, back_margin=back_margin_no_drop_day,
                                                   num_of_days=drop_date_set_list_training_dataset.shape[0]*ratio_of_label0_to_label1)

    # normalization
    _, training_database_set_norm = preprocessing_dataframe_minmaxscaler(training_database_set, [])

    # generating of dataset with label 1 and label 0

    # print("training_database_set_norm.shape)", training_database_set_norm.shape)

    trainX_label1, trainY_label1 = generate_dataset(features_set=training_database_set_norm, the_day_set=drop_date_set_list_training_dataset, label_is_drop=1,
                             range_of_days_before_the_day=range_of_days_before_the_drop_day, adding_one_padding=ADD_ONE_PADDING)
    #
    # print("X_label=1", trainX_label1.shape)
    # print("Y_label=1", trainY_label1.shape)

    trainX_label0, trainY_label0 = generate_dataset(features_set=training_database_set_norm, the_day_set=date_normal_set_list_training_dataset, label_is_drop=0,
                             range_of_days_before_the_day=range_of_days_before_the_drop_day, adding_one_padding=ADD_ONE_PADDING)
    #
    # print("X_label=0", trainX_label0.shape)
    # print("Y_label=0", trainY_label0.shape)

    ############

    # data augmentation
    # ratio between number of trainX_label1 and trainX_label0
    if if_data_augmentation:
        # print("trainX_label0", trainX_label0.shape)
        # print("trainX_label1", trainX_label1.shape)
        ratio_between_label1_and_label0 = int(trainX_label0.shape[0]/trainX_label1.shape[0])
        trainX_label1_aug = pre_data_augmentation(trainX_label1, ratio_between_label1_and_label0-1, data_augmentation_method)
        trainX_label1 = np.concatenate((trainX_label1, trainX_label1_aug), axis=0)
        # print("trainX_label1", trainX_label1.shape)
        # print("trainY_label1", trainY_label1.shape)
        trainY_label1 = np.array([[0, 1]*trainX_label1.shape[0]]).reshape(trainX_label1.shape[0], 2)
        # print("trainY_label1", trainY_label1.shape)

    if balanced_training_dataset:
        trainX_label0, trainY_label0, trainX_label1, trainY_label1 = balance_dataset(trainX_label0, trainY_label0, trainX_label1, trainY_label1)

    # print("Training data:")
    # print("trainX_label0", trainX_label0.shape)
    # print("trainX_label1", trainX_label1.shape)
    # print("trainY_label0", trainY_label0.shape)
    # print("trainY_label1", trainY_label1.shape)

    trainX = np.concatenate((trainX_label1, trainX_label0), axis=0)
    trainY = np.concatenate((trainY_label1, trainY_label0), axis=0)

    del trainX_label1, trainX_label0, trainY_label1, trainY_label0
    gc.collect()

    # Shuffle
    state = np.random.get_state()
    np.random.shuffle(trainX)
    np.random.set_state(state)
    np.random.shuffle(trainY)


    trainY_label = from_one_hot_to_1d_array(trainY)
    testY_label = from_one_hot_to_1d_array(testY)
    print("trainX", trainX.shape)
    print("trainY", trainY.shape)
    # print("trainY_label", trainY_label)
    # pd.DataFrame(trainY).to_csv("trainY_20210723.csv")
    # pd.DataFrame(trainY_label).to_csv("trainY_label_20210723.csv")
    # print("trainY_label", trainY_label.shape)
    print("testX", testX.shape)
    print("testY", testY.shape)
    # print("testY_label", testY_label.shape)

    # trainX = trainX[:, :, 5:]
    # testX = testX[:, :, 5:]

    # with h5py.File('testX_testY.h5', 'a') as hf:
    #      hf.create_dataset("testX",  data=testX)
    #      hf.create_dataset("testY", data=testY)

    model_sa = selfAttentionDjia(hidden=42, n_layers=8, attn_heads=6, dropout=0.1, train_method=TRAIN_METHOD,
                              target_method=TARGET_METHOD)
    model_sa = train(model_sa, trainX, trainY_label, patience, valid_ratio, learning_rate, epoch_num,
                     model_save_name="classifier_20210723",
                     train_method=TRAIN_METHOD, target_method=TARGET_METHOD, pre_trained_model_file="./model_save/regression_last_one.pth")

    # del trainX, trainY, trainY_label
    # gc.collect()



    # predY = pred_with_model(model_sa, trainX)
    predY = pred_with_model(model_sa, testX)
    del testX
    gc.collect()
    # print("predY.shape", predY.shape)

    predY = predY.cpu().numpy()
    # print("predY.shape", predY.shape)

    predY_label = from_one_hot_to_1d_array(predY)

    trueY, trueY_label = testY, testY_label

    # print("predY_degree_of_Truth\n", np.around(predY.T[1], 2))
    # print("predY_label\n", predY_label)
    # print("trueY_label\n", trueY_label)
    #
    # print("predY_degree_of_Truth\n", np.around(predY.T[1], 2).shape)
    # print("predY_label\n", predY_label.shape)
    # print("trueY_label\n", trueY_label.shape)

    # pd.DataFrame(predY).to_csv("predY_20210723.csv")
    # pd.DataFrame(trueY).to_csv("trueY_20210723.csv")
    # pd.DataFrame(predY_label).to_csv("predY_label_20210723.csv")
    # pd.DataFrame(trueY_label).to_csv("trueY_label_20210723.csv")

    # combined_result = np.concatenate((np.around(predY.T[1], 2), predY_label, trueY_label), axis=0).reshape(3, -1)
    # print(combined_result.shape)
    # np.savetxt("temp_combined_result_degree_of_truth.csv", combined_result, delimiter=",")

    # acc = metrics.accuracy_score(trueY_label, predY_label)
    # tn, fp, fn, tp = metrics.confusion_matrix(trueY_label, predY_label).ravel()
    acc = metrics.accuracy_score(trueY_label, predY_label)
    tn, fp, fn, tp = metrics.confusion_matrix(trueY_label, predY_label).ravel()
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    # a = metrics.confusion_matrix(trueY_label, predY_label).ravel()
    # print("a", a)
    tpr = tp / (tp + fn)
    ppv = tp / (tp + fp)
    auc = metrics.roc_auc_score(trueY, predY)
    f1 = metrics.f1_score(predY_label, trueY_label)

    print("accuracy: %.4f" % acc)
    print("tpr:%.4f" % tpr)
    print("ppv:%.4f" % ppv)
    print("auc:%.4f" % auc)
    print("f1-score:%.4f" % f1)

