from selfattention_main import *

if __name__ == '__main__':
    # hyper-parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = 10
    valid_ratio = 0.15
    learning_rate = 1e-3
    epoch_num = 100
    # different train method and target method
    TRAIN_METHOD = "regression"  # classification, regression
    TARGET_METHOD = "last_one"  # last_one, average, padding_one
    # use padding as last one to do the classification
    ADD_ONE_PADDING = True if TARGET_METHOD == "padding_one" else False
    MORE_DATA_RATIO = 5
    data_augmentation_method = "mixed"

    print("torch.cuda.is_available()", torch.cuda.is_available())

    with h5py.File('training_data_regression.h5', 'r') as hf:
        dataset_500_X = hf['training_data_regression_100_X'][:]
        dataset_500_Y = hf['training_data_regression_100_Y'][:]

    print("dataset_500_X", dataset_500_X.shape)
    print("dataset_500_Y", dataset_500_Y.shape)

    more_data_ratio = MORE_DATA_RATIO
    dataset_500_X_aug = pre_data_augmentation(dataset_500_X, more_data_ratio-1, data_augmentation_method)
    dataset_500_X_aug = np.concatenate((dataset_500_X, dataset_500_X_aug), axis=0)
    # print("dataset_500_X_aug", dataset_500_X_aug.shape)

    dataset_500_Y_aug = dataset_500_Y
    for _ in range(more_data_ratio-1):
        dataset_500_Y_aug = np.concatenate((dataset_500_Y_aug, dataset_500_Y), axis=0)
    # print("dataset_500_Y_aug", dataset_500_Y_aug.shape)

    # print("trainX_label1", trainX_label1.shape)
    # print("trainY_label1", trainY_label1.shape)
    # trainY_label1 = np.array([[0, 1] * trainX_label1.shape[0]]).reshape(trainX_label1.shape[0], 2)
    # print("trainY_label1", trainY_label1.shape)

    trainX_500, testX_500 = separate_training_dataset_and_testing_dataset(dataset_500_X_aug, 0.8)
    trainY_500, testY_500 = separate_training_dataset_and_testing_dataset(dataset_500_Y_aug, 0.8)

    print("trainX", trainX_500.shape)
    print("testX", testX_500.shape)
    print("trainY", trainY_500.shape)
    print("testY", testY_500.shape)

    model_reg = selfAttentionDjia(hidden=42, n_layers=8, attn_heads=6, dropout=0.1, train_method=TRAIN_METHOD,
                                 target_method=TARGET_METHOD)

    model_reg = train(model_reg, trainX_500, trainY_500, patience, valid_ratio, learning_rate, epoch_num,
                      model_save_name="regressor_20210723",
                     train_method=TRAIN_METHOD, target_method=TARGET_METHOD)

    del trainX_500, trainY_500
    gc.collect()

    # predY = pred_with_model(model_sa, trainX)
    predY = pred_with_model(model_reg, testX_500)
    del testX_500
    gc.collect()
    # print("predY.shape", predY.shape)

    predY = predY.cpu().numpy()
    # print("predY.shape", predY.shape)

    MSE = metrics.mean_squared_error(testY_500, predY)

    print("MSE", MSE)
    #
    # predY_label = from_one_hot_to_1d_array(predY)

    # trueY = testY_500

