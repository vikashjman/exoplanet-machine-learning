import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import json
import random
import numpy as np
import os


class LightFluxProcessor:
    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True, smote=True,
                 polynomial_features=False, pca=False, variance_threshold=False, savgol_filtering=False):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize
        self.smote = smote
        self.polynomial_features = polynomial_features
        self.pca = pca
        self.variance_threshold = variance_threshold
        self.savgol_filtering = savgol_filtering

    def fourier_transform(self, X):
        return np.abs(np.fft.fft(X, n=X.size))[:X.size // 2]

    def normalize_data(self, X):
        return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

    def apply_gaussian(self, X, sigma=1.0):
        return np.array([gaussian_filter(x, sigma=sigma) for x in X])

    def apply_savgol(self, X, window_length=5, polyorder=2):
        return np.array([savgol_filter(x, window_length=window_length, polyorder=polyorder) for x in X])

    def standardize_data(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def apply_pca(self, X, n_components=0.95):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)

    def apply_variance_threshold(self, X, threshold=0.01):
        selector = VarianceThreshold(threshold=threshold)
        return selector.fit_transform(X)

    def expand_polynomial_features(self, X, degree=2):
        poly = PolynomialFeatures(degree=degree)
        return poly.fit_transform(X)

    def process(self, df_train_x, df_dev_x):
        if self.fourier:
            print("Applying Fourier transform...")
            df_train_x = np.apply_along_axis(self.fourier_transform, axis=1, arr=df_train_x)
            df_dev_x = np.apply_along_axis(self.fourier_transform, axis=1, arr=df_dev_x)

        if self.normalize:
            print("Normalizing data...")
            df_train_x = self.normalize_data(df_train_x)
            df_dev_x = self.normalize_data(df_dev_x)

        if self.gaussian:
            print("Applying Gaussian smoothing...")
            df_train_x = self.apply_gaussian(df_train_x)
            df_dev_x = self.apply_gaussian(df_dev_x)

        if self.savgol_filtering:
            print("Applying Savitzky-Golay filtering...")
            df_train_x = self.apply_savgol(df_train_x)
            df_dev_x = self.apply_savgol(df_dev_x)

        if self.standardize:
            print("Standardizing data...")
            df_train_x = self.standardize_data(df_train_x)
            df_dev_x = self.standardize_data(df_dev_x)

        if self.polynomial_features:
            print("Expanding polynomial features...")
            df_train_x = self.expand_polynomial_features(df_train_x)
            df_dev_x = self.expand_polynomial_features(df_dev_x)

        if self.pca:
            print("Applying PCA for dimensionality reduction...")
            df_train_x = self.apply_pca(df_train_x)
            df_dev_x = self.apply_pca(df_dev_x)

        if self.variance_threshold:
            print("Applying variance threshold for feature selection...")
            df_train_x = self.apply_variance_threshold(df_train_x)
            df_dev_x = self.apply_variance_threshold(df_dev_x)

        print("Finished Processing!")
        return df_train_x, df_dev_x

    def apply_smote(self, X_train, Y_train):
        if self.smote:
            print("Applying SMOTE...")
            smote = SMOTE()
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
        return X_train, Y_train






