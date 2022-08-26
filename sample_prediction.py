import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from featurize import features_for_mic_group, get_concat_fv


def main():
    # Load the necessary data files
    df = pd.read_pickle("../data/featurized_data/dov_data.pkl")
    df["concatenated_fv"] = df["concatenated_fv"] + df["reverb_feat"]
    feat = "concatenated_fv" # Contains the full-length feature vector

    # Train an eight-way classifier for each rotation angle
    X_train = df["concatenated_fv"].tolist()
    Y_train = df["rotation"].tolist()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = np.nan_to_num(X_train)
    Y_train = Y_train.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    clf = ExtraTreesClassifier(n_estimators=1000)
    clf.fit(X_train,Y_train)

    # Predict the direction of voice from a sample recording

    MIC_GROUP = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    fs = 48000
    c_size = None
    avg_channels = False

    folder = "../data/example_wav/180/"
    filename = "channel"
    wavs, features_gcc, features_xcorr_gcc, acorr_gcc, features_per_wav, extended_fv, autocorr_feat, reverb_sl = features_for_mic_group(folder, filename, fs, MIC_GROUP, c_size, avg_channels)
    fv = get_concat_fv(features_gcc, autocorr_feat, features_per_wav, extended_fv, avg_channels)
    test_fv = [fv + reverb_sl]
    X_test = np.array(test_fv)
    X_test = np.nan_to_num(X_test)
    X_test = min_max_scaler.transform(X_test)
    rot_pred = clf.predict(X_test)[0]

    print("Predicted direction of voice is", rot_pred, "degrees")

if __name__ == "__main__":
    main()

