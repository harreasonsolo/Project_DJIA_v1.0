from base_utils_and_data_generator import *
import tensorflow_addons as tfa

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
    ratio_matrix = [[15, 0.06]]
    range_of_days_before_the_drop_day = [-200, 0]
    length_of_days_before_the_drop_day = range_of_days_before_the_drop_day[1] - range_of_days_before_the_drop_day[0]
    #number_of_days_between_two_dropping_date = 7
    number_of_days_between_two_dropping_date = 0
    # front_margin_no_drop_day, back_margin_no_drop_day = 30, 60
    front_margin_no_drop_day, back_margin_no_drop_day = 0, 0
    ratio_of_label0_to_label1 = 8
    epoch_num = 80
    #balanced_testing_dataset = False
    balanced_training_dataset = True
    if_data_augmentation = True
    data_augmentation_method = "mixed"
    # data_augmentation_method can be ["jitter", "scaling", "magnitude_warp", "window_warp", "mixed"]
    ADD_ONE_PADDING = False


# DNN
# Traditional RNN
def traditional_rnn_model(number_of_days, number_of_features):
    model = tf.keras.models.Sequential()
    # rnn layers
    # model.add(tf.keras.layers.SimpleRNN(64, return_sequences=True, activation='relu',
    #                                    input_shape=(number_of_days, number_of_features)))

    model.add(tf.keras.layers.RNN(tfa.rnn.LayerNormSimpleRNNCell(64), return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.RNN(tfa.rnn.LayerNormSimpleRNNCell(64), return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.RNN(tfa.rnn.LayerNormSimpleRNNCell(64), return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.RNN(tfa.rnn.LayerNormSimpleRNNCell(64),
                                  input_shape=(number_of_days, number_of_features)))

    # dense layer
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    model.add(tf.keras.layers.Dense(16, activation='tanh'))
    # classifier
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    # return the constructed network architecture
    return model


def lstm_model(number_of_days, number_of_features):
    model = tf.keras.models.Sequential()
    # LSTM layers
    model.add(tf.keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(64), return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(64), return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(64), return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(64),
                                  input_shape=(number_of_days, number_of_features)))
    # dense layer
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    model.add(tf.keras.layers.Dense(16, activation='tanh'))
    # classifier
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    # return the constructed network architecture
    return model


def gru_model(number_of_days, number_of_features):
    model = tf.keras.models.Sequential()
    # GRU layers
    model.add(tf.keras.layers.GRU(64, return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.LayerNormalization(axis=1))
    model.add(tf.keras.layers.GRU(64, return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.LayerNormalization(axis=1))
    model.add(tf.keras.layers.GRU(64, return_sequences=True,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.LayerNormalization(axis=1))
    model.add(tf.keras.layers.GRU(64,
                                  input_shape=(number_of_days, number_of_features)))
    model.add(tf.keras.layers.LayerNormalization(axis=1))
    # dense layer
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    model.add(tf.keras.layers.Dense(16, activation='tanh'))
    # classifier
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    # return the constructed network architecture
    return model



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
    unemployment_rate.to_csv('temp_concat_df_before_binning.csv')
    unemployment_rate_binning.to_csv('temp_concat_df_binning.csv')

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
    #
    # Start to separate the training data and testing data
    training_database_set, testing_database_set = separate_training_dataset_and_testing_dataset(temp_concat, 0.67)

    # 1. generating testing dataset
    # 1.1 normalization using scalar fit with training dataset
    minmax_scalar_training_dataset, _ = preprocessing_dataframe_minmaxscaler(training_database_set, [])
    _, testing_database_set_norm = preprocessing_dataframe_minmaxscaler(testing_database_set, minmax_scalar_training_dataset)
    testing_database_set = testing_database_set_norm

    window_size = length_of_days_before_the_drop_day
    start_seq = len(training_database_set)
    testing_dataset, date_array, date_price_array = time_series_data_generator_with_window(testing_database_set, [-window_size, 0], window_size, 10000, adding_one_padding=ADD_ONE_PADDING)
    # drop_date_set_list = there_is_a_sudden_drop(DJIA_dataframe=testing_database_set[start_datetime+timedelta(days=length_of_days_before_the_drop_day*8/5):end_datetime],
    #                                             drop_time_amp_ratio_matrix=ratio_matrix,
    #                                             number_of_days_between_two_dropping_date=number_of_days_between_two_dropping_date)
    drop_date_set_list_testing_dataset = there_is_a_sudden_drop(DJIA_dataframe=testing_database_set[length_of_days_before_the_drop_day:],
                                                drop_time_amp_ratio_matrix=ratio_matrix,
                                                number_of_days_between_two_dropping_date=number_of_days_between_two_dropping_date)
    drop_date_set_list_testing_dataset.to_csv('drop_date_set_list_testing_dataset.csv')

    testX, testY = generate_dataset_for_testing_dataset(testing_database_set[length_of_days_before_the_drop_day:], drop_date_set_list_testing_dataset, testing_dataset)
    print("testX.shape", testX.shape)
    #print(testX[:3])
    print("testY.shape", testY.shape)
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
    # model_scaler_ten_year_US_Government_Bond_Yields, df_ten_year_US_Government_Bond_Yields = preprocessing_dataframe_minmaxscaler(ten_year_US_Government_Bond_Yields)
    # df_ten_year_US_Government_Bond_Yields.to_csv('temp_concat_df_nor_ten_year_US_Government_Bond_Yields.csv')
    _, training_database_set_norm = preprocessing_dataframe_minmaxscaler(training_database_set, [])

    # generating of dataset with label 1 and label 0

    print("training_database_set_norm.shape)", training_database_set_norm.shape)

    trainX_label1, trainY_label1 = generate_dataset(features_set=training_database_set_norm, the_day_set=drop_date_set_list_training_dataset, label_is_drop=1,
                             range_of_days_before_the_day=range_of_days_before_the_drop_day, adding_one_padding=ADD_ONE_PADDING)

    trainX_label0, trainY_label0 = generate_dataset(features_set=training_database_set_norm, the_day_set=date_normal_set_list_training_dataset, label_is_drop=0,
                             range_of_days_before_the_day=range_of_days_before_the_drop_day, adding_one_padding=ADD_ONE_PADDING)

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

    print("trainX_label0", trainX_label0.shape)
    print("trainX_label1", trainX_label1.shape)
    print("trainY_label0", trainY_label0.shape)
    print("trainY_label1", trainY_label1.shape)

    trainX = np.concatenate((trainX_label1, trainX_label0), axis=0)
    trainY = np.concatenate((trainY_label1, trainY_label0), axis=0)

    print("trainX", trainX.shape)
    print("trainY", trainY.shape)

    # Shuffle
    state = np.random.get_state()
    np.random.shuffle(trainX)
    np.random.set_state(state)
    np.random.shuffle(trainY)

    trainX, validX, trainY, validY = model_selection.train_test_split \
        (trainX, trainY, test_size=0.1, random_state=42)
    validset = (validX, validY)

    # early stopping criteria
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # use validation accuracy for stopping
        min_delta=0.0001, patience=5,
        verbose=1, mode='auto')
    callbacks_list = [earlystop]

    # gru_model
    # lstm_model
    # traditional_rnn_model

    # rnn_model = traditional_rnn_model(length_of_days_before_the_drop_day, temp_concat.shape[1])
    rnn_model = gru_model(length_of_days_before_the_drop_day, temp_concat.shape[1])

    rnn_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    history = rnn_model.fit(trainX, trainY, epochs=epoch_num, batch_size=50,
                            callbacks=callbacks_list,
                            validation_data=validset,  # specify the validation set
                            verbose=2)

    # LSTM
    predY = rnn_model.predict(testX, verbose=False)
    predY_label = tf.math.argmax(predY, 1)
    trueY_label = tf.math.argmax(testY, 1)
    print("predY_label", predY_label)
    print("trueY_label", trueY_label)

    acc = metrics.accuracy_score(trueY_label, predY_label)
    tn, fp, fn, tp = metrics.confusion_matrix(trueY_label, predY_label).ravel()
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    tpr = tp/(tp+fn)
    ppv = tp/(tp+fp)
    # print("testY", testY)
    # print("predY", predY)
    auc = metrics.roc_auc_score(testY, predY)
    f1 = metrics.f1_score(predY_label, trueY_label)

    plot_result(date_array, date_price_array, np.around(predY.T[1], 2))

    print("accuracy: %.4f" % acc)
    print("tpr:%.4f" % tpr)
    print("ppv:%.4f" % ppv)
    print("auc:%.4f" % auc)
    print("f1-score:%.4f" % f1)