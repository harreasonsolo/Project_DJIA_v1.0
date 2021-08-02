from selfattention_main import *
import h5py

if __name__ == "__main__":

    TRAIN_METHOD = "classification"  # classification, regression
    TARGET_METHOD = "last_one"  # last_one, average, padding_one

    with h5py.File('testX_testY.h5', 'r') as hf:
        testX = hf['testX'][:]
        testY = hf['testY'][:]

    model_sa = selfAttentionDjia(hidden=42, n_layers=8, attn_heads=6, dropout=0.1, train_method=TRAIN_METHOD,
                                  target_method=TARGET_METHOD)

    pthfile = '.\model_save\classifier_20210723_classification_last_one.pth'
    model_sa.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))

    predY = pred_with_model(model_sa, testX)

    del testX
    gc.collect()
    print("predY.shape", predY.shape)

    predY = predY.cpu().numpy()
    print("predY.shape", predY.shape)

    testY_label = from_one_hot_to_1d_array(testY)
    predY_label = from_one_hot_to_1d_array(predY)

    trueY, trueY_label = testY, testY_label

    print("predY_degree_of_Truth\n", np.around(predY.T[1], 2))
    print("predY_label\n", predY_label)
    print("trueY_label\n", trueY_label)

    print("predY_degree_of_Truth\n", np.around(predY.T[1], 2).shape)
    print("predY_label\n", predY_label.shape)
    print("trueY_label\n", trueY_label.shape)

    acc = metrics.accuracy_score(trainY_label, predY_label)
    tn, fp, fn, tp = metrics.confusion_matrix(trainY_label, predY_label).ravel()
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