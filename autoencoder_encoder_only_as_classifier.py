import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from utils.pytorchtools import EarlyStopping
from torch.utils.data import SubsetRandomSampler
import numpy as np
import h5py
from sklearn import *
from tqdm import tqdm
# from rnn_with_utils import *
from base_utils_and_data_generator import *
import gc

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
    range_of_days_before_the_drop_day = [-1095, 0]
    #range_of_days_before_the_drop_day = [-180, 0]
    length_of_days_before_the_drop_day = range_of_days_before_the_drop_day[1] - range_of_days_before_the_drop_day[0]
    #number_of_days_between_two_dropping_date = 30
    number_of_days_between_two_dropping_date = 0
    # front_margin_no_drop_day, back_margin_no_drop_day = 30, 60
    front_margin_no_drop_day, back_margin_no_drop_day = 0, 0
    ratio_of_label0_to_label1 = 8
    # balanced_testing_dataset = False
    balanced_training_dataset = True
    if_data_augmentation = False
    data_augmentation_method = "mixed"
    # data_augmentation_method can be ["jitter", "scaling", "magnitude_warp", "window_warp", "mixed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = 10
    valid_ratio = 0.2

def get_hidden_representation_from_encoder(model, input):
    # this input can be the augmentation of -1 *(365*3*36) data
    model.eval()
    with torch.no_grad():
        output = model.encoder(input)
    return output



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #nn.Linear(365 * 3 * 36, 6570),
            nn.Linear(365 * 3 * 42, 6570),
            nn.LayerNorm(6570, elementwise_affine=False),
            nn.ReLU(True),
            nn.Linear(6570, 1024),
            nn.LayerNorm(1024, elementwise_affine=False),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.LayerNorm(256, elementwise_affine=False),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LayerNorm(1024, elementwise_affine=False),
            nn.ReLU(True),
            nn.Linear(1024, 6570),
            nn.LayerNorm(6570, elementwise_affine=False),
            nn.ReLU(True),
            #nn.Linear(6570, 365 * 3 * 36),
            nn.Linear(6570, 365 * 3 * 42),
            # nn.LayerNorm(365 * 3 * 42),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class fc_classifier(nn.Module):
    def __init__(self):
        super(fc_classifier, self).__init__()
        self.dense_layer_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.dense_layer_classifier(x)
        return output


def train_classifier_for_hidden_representation(trainX, trainY, patience, valid_ratio):
    # initial the model
    model = fc_classifier()
    model.to(device)
    # setting the early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # initial the training dataset and validation dataset
    #     trainX = torch.Tensor(trainX)
    #     trainY = torch.Tensor(trainY)
    training_dataset = Data.TensorDataset(trainX, trainY)
    #data_loader = Data.DataLoader(training_dataset, batch_size=50, shuffle=True)

    # obtain training indices that will be used for validation
    num_train = len(training_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(valid_ratio * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(training_dataset,
                                               batch_size=50,
                                               sampler=train_sampler,
                                               num_workers=2)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(training_dataset,
                                               batch_size=50,
                                               sampler=valid_sampler,
                                               num_workers=2)

    # initial the loss function
    criterion = nn.CrossEntropyLoss()

    # initial the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    for epoch in tqdm(range(epoch_num)):
        # loss array
        train_loss_array = []
        valid_loss_array = []

        # train
        model.train()
        for train_X, train_Y in train_loader:
            train_X = train_X.to(device)
            train_Y = train_Y.to(device, dtype=torch.int64)
            # fwd
            output = model(train_X)
            loss = criterion(output, train_Y)
            # bp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.data)
        # print("train_loss_array", len(train_loss_array))
        # validation
        model.eval()
        for valid_X, valid_Y in valid_loader:
            valid_X = valid_X.to(device)
            valid_Y = valid_Y.to(device, dtype=torch.int64)
            output = model(valid_X)
            loss = criterion(output, valid_Y)
            valid_loss_array.append(loss.data)
        # print("valid_loss_array", len(valid_loss_array))
        print("epoch:[{}/{}], training loss:{:.4f}, validation loss:{:.4f}".format(epoch + 1, epoch_num, np.array(train_loss_array).mean(), np.array(valid_loss_array).mean()))

        early_stopping(np.array(valid_loss_array).mean(), model, save_model=False)

        if early_stopping.early_stop:
            print("early stop.")
            break

    # print("train_loss_array", train_loss_array)
    # print("train_loss_array", len(train_loss_array))
    # print("valid_loss_array", valid_loss_array)
    # print("valid_loss_array", len(valid_loss_array))

    torch.save(model.state_dict(), "./fc_classifier.pth")
    return model


def pred_with_fc_classifer(model, input):
    input = input.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input)
    return output


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
    # dataset_df_list = [DJIA, SP500, ten_year_US_Government_Bond_Yields, ten_Year_Treasury_Constant_Maturity_Rate, CPI_PPI_PCE_US,
    #                    Effective_Federal_Funds_Rate, GDP_of_US, Gold_Price, Interest_Rates_Discount_Rate_US, NASDAQ_Composite,
    #                    Russel_2000, unemployment_rate_binning]
    # dataset_name_list = ["DJIA", "SP500", "ten_year_US_Government_Bond_Yields", "ten_Year_Treasury_Constant_Maturity_Rate", "CPI_PPI_PCE_US",
    #                      "Effective_Federal_Funds_Rate", "GDP_of_US", "Gold_Price", "Interest_Rates_Discount_Rate_US", "NASDAQ_Composite",
    #                      "Russel_2000", "unemployment_rate_binning"]
    dataset_df_list = [DJIA, SP500, ten_year_US_Government_Bond_Yields, ten_Year_Treasury_Constant_Maturity_Rate, CPI_PPI_PCE_US,
                       Effective_Federal_Funds_Rate, GDP_of_US, Gold_Price, Interest_Rates_Discount_Rate_US, NASDAQ_Composite,
                       Russel_2000, unemployment_rate_binning]
    dataset_name_list = ["DJIA", "SP500", "ten_year_US_Government_Bond_Yields", "ten_Year_Treasury_Constant_Maturity_Rate", "CPI_PPI_PCE_US",
                         "Effective_Federal_Funds_Rate", "GDP_of_US", "Gold_Price", "Interest_Rates_Discount_Rate_US", "NASDAQ_Composite",
                         "Russel_2000", "unemployment_rate_binning"]
    temp_concat, start_datetime, end_datetime = concat_dataset(dataset_df_list, dataset_name_list)
    # DJIA.to_csv('DJIA.csv')
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
    testing_dataset, date_array, date_price_array = time_series_data_generator_with_window(testing_database_set, [-window_size, 0], window_size, 10000)
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
                             range_of_days_before_the_day=range_of_days_before_the_drop_day)
    #
    # print("X_label=1", trainX_label1.shape)
    # print("Y_label=1", trainY_label1.shape)

    trainX_label0, trainY_label0 = generate_dataset(features_set=training_database_set_norm, the_day_set=date_normal_set_list_training_dataset, label_is_drop=0,
                             range_of_days_before_the_day=range_of_days_before_the_drop_day)
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

    print("trainX", trainX.shape)
    print("trainY", trainY.shape)
    print("testX", testX.shape)
    print("testY", testY.shape)

    # trainX = trainX[:, :, 5:]
    # testX = testX[:, :, 5:]

    # print("trainX", trainX.shape)
    # print("trainY", trainY.shape)
    # print("testX", testX.shape)
    # print("testY", testY.shape)

    #print("trainX[0]", trainX[0][0])

    ###############################################
    # This data should be replaced by real trainX, trainY, testX, testY
    # with h5py.File('../input/ae-training-data/ae_data.h5', 'r') as hf:
    #     trainX = hf['training_data_1095'][:150]
    # trainY = np.random.randint(2, size=(150, 2))
    # testX = trainX.copy()
    # testY = trainY.copy()

    trainY_label = from_one_hot_to_1d_array(trainY)
    testY_label = from_one_hot_to_1d_array(testY)
    ###############################################

    model_ae = autoencoder()
    model_ae.load_state_dict(torch.load("model_save/ae_model_1095to256_addingDJIA_only_trainingset_replace relu with sigmoid in encoder.pth", map_location=torch.device('cpu')))

    learning_rate = 1e-3
    epoch_num = 50

    # step1: get trainX_hidden from trainX with Encoder
    trainX = torch.Tensor(trainX)
    #print("trainX.shape", trainX.shape)
    trainX = trainX.view(trainX.size(0), -1)
    trainX_hidden = get_hidden_representation_from_encoder(model_ae, trainX)
    #print("trainX_hidden.shape", trainX_hidden.shape)
    #print("trainX_hidden.type()", trainX_hidden.type())

    # step2: use trainX_hidden and trainY to train the model of FC Classifier
    trainY_label = torch.Tensor(trainY_label)
    #     training_dataset = Data.TensorDataset(trainX_hidden, trainY)
    #     dataset_loader = Data.DataLoader(training_dataset, batch_size=50, shuffle=True, num_workers=2)
    model_fc = train_classifier_for_hidden_representation(trainX_hidden, trainY_label, patience, valid_ratio)

    # step3: evaluate the model with testing dataset
    # 3.1 get testX_hidden from testX with Encoder
    testX = torch.Tensor(testX)
    testX = testX.view(testX.size(0), -1)
    testX_hidden = get_hidden_representation_from_encoder(model_ae, testX)
    #print("testX_hidden.shape", testX_hidden.shape)
    #print("testX_hidden.type()", testX_hidden.type())
    # 3.2 get testY from testX with FC_Classifier
    testX_hidden = testX_hidden.view(testX_hidden.size(0), -1)
    predY = pred_with_fc_classifer(model_fc, testX_hidden)
    #print("predY.shape", predY.shape)

    predY = predY.cpu().numpy()
    #print("predY.shape", predY.shape)

    trueY = testY
    trueY_label = testY_label
    predY_label = from_one_hot_to_1d_array(predY)

    # print("predY_degree_of_Truth\n", np.around(predY.T[1],2))
    # print("predY_label\n", predY_label)
    # print("trueY_label\n", trueY_label)
    #
    # print("predY_degree_of_Truth\n", np.around(predY.T[1],2).shape)
    # print("predY_label\n", predY_label.shape)
    # print("trueY_label\n", trueY_label.shape)

    combined_result = np.concatenate((np.around(predY.T[1], 2), predY_label, trueY_label), axis=0).reshape(3, -1)
    #print(combined_result.shape)
    np.savetxt("temp_combined_result_degree_of_truth.csv", combined_result, delimiter=",")

    acc = metrics.accuracy_score(trueY_label, predY_label)
    tn, fp, fn, tp = metrics.confusion_matrix(trueY_label, predY_label).ravel()
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    # a = metrics.confusion_matrix(trueY_label, predY_label).ravel()
    # print("a", a)
    tpr = tp / (tp + fn)
    ppv = tp / (tp + fp)
    auc = metrics.roc_auc_score(trueY, predY)
    f1 = metrics.f1_score(predY_label, trueY_label)

    plot_result(date_array, date_price_array, np.around(predY.T[1],2))

    print("accuracy: %.4f" % acc)
    print("tpr:%.4f" % tpr)
    print("ppv:%.4f" % ppv)
    print("auc:%.4f" % auc)
    print("f1-score:%.4f" % f1)
