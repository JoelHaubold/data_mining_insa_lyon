import pandas as pd
import scipy
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn import neighbors
from pathlib import Path

data_file = Path("data/DSL-StrongPasswordData.csv")
t = 2.3

if __name__ == '__main__':
    df_data = pd.read_csv(data_file)
    print(df_data)
    subjects = df_data["subject"].unique()
    s_dists = []
    i_dists = []
    for s in subjects:
        s_data = df_data[df_data["subject"] == s]
        i_data = df_data[~(df_data["subject"] == s)]
        train_data = s_data[s_data["sessionIndex"] <= 4].drop(["subject", "sessionIndex", "rep"], axis=1)
        test_s_data = s_data[s_data["sessionIndex"] > 4].drop(["subject", "sessionIndex", "rep"], axis=1)
        test_i_data = i_data[(i_data["sessionIndex"] == 1) & (i_data["rep"] <= 5)].drop(["subject", "sessionIndex", "rep"], axis=1)
        dist = DistanceMetric.get_metric('manhattan') #TODO: Scaled manhattan
        # cov = test_s_data.cov()
        # dist = DistanceMetric.get_metric('mahalanobis', VI=cov)

        model = train_data.mean().to_frame().transpose()
        # print(model.shape)
        # print(model)
        # print(test_s_data.shape)
        s_dist = dist.pairwise(model, test_s_data)[0]
        i_dist = dist.pairwise(model, test_i_data)[0]
        s_dists.extend(s_dist)
        i_dists.extend(i_dist)
    s_dists = np.array(s_dists)
    i_dists = np.array(i_dists)
    print(np.nanmean(s_dists))
    print(np.nanmean(i_dists))
    print(np.nanvar(s_dists))
    print(np.nanvar(i_dists))
    print(len(s_dists))
    print(len(i_dists))

    print((s_dists <= t).sum()/len(s_dists))
    print((i_dists > t).sum()/len(i_dists))

