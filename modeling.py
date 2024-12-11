import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from mordred import Calculator, descriptors
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


def compute_mordred(df):
    """
    Compute the Mordred descriptors for a DataFrame containing SMILES strings.

    Parameters:
    -----------
    - df (pandas.DataFrame): A DataFrame containing a column named "SMILES" with SMILES strings.

    Returns:
    --------
    - pandas.DataFrame: A DataFrame with the original data (excluding the "SMILES" column) and the computed Mordred descriptors.

    Notes:
    ------
    - The function computes Mordred descriptors for each molecule represented by a SMILES string.
    - It removes any constant columns (columns with the same value in all rows).
    - It also removes any columns containing NaN values.
    - The resulting DataFrame does not include the "SMILES" column.
    """

    data = df.copy()

    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in data["SMILES"]]
    descrs = calc.pandas(mols).astype(float)

    data = pd.concat([data, descrs], axis=1)
    data = data.loc[:, (data != data.iloc[0]).any()]  ## remove all constant columns
    data.dropna(
        axis=1, how="any", inplace=True
    )  ## remove any columns containing NaN values
    data.drop(columns=["SMILES"], inplace=True)

    return data


def compute_rdkit(df):
    """
    Compute the RDKit descriptors for a DataFrame containing SMILES strings.

    Parameters:
    -----------
    - df (pandas.DataFrame): A DataFrame containing a column named "SMILES" with SMILES strings.

    Returns:
    --------
    - pandas.DataFrame: A DataFrame with the original data (excluding the "SMILES" column) and the computed RDKit descriptors.

    Notes:
    ------
    - The function computes RDKit descriptors for each molecule represented by a SMILES string.
    - It removes any constant columns (columns with the same value in all rows).
    - It also removes any columns containing NaN values.
    - The resulting DataFrame does not include the "SMILES" column.
    """

    data = df.copy()

    mols = [Chem.MolFromSmiles(smi) for smi in data["SMILES"]]
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    descrs = pd.DataFrame(descrs)

    data = pd.concat([data, descrs], axis=1)
    data = data.loc[:, (data != data.iloc[0]).any()]
    data.dropna(axis=1, how="any", inplace=True)
    data.drop(columns=["SMILES"], inplace=True)

    return data


def compute_morgan(df):
    """
    Compute the Morgan fingerprints for a DataFrame containing SMILES strings.

    Parameters:
    -----------
    - df (pandas.DataFrame): A DataFrame containing a column named "SMILES" with SMILES strings.

    Returns:
    --------
    - pandas.DataFrame: A DataFrame with the original data (excluding the "SMILES" column) and the computed Morgan fingerprints.

    Notes:
    ------
    - The function computes Morgan fingerprints for each molecule represented by a SMILES string.
    - It creates a new column for each unique fingerprint bit and counts the occurrences of each bit for each molecule.
    - It fills any missing values with 0.
    - It removes any constant columns (columns with the same value in all rows).
    - The resulting DataFrame does not include the "SMILES" column.
    """

    data = df.copy()
    new_columns_data = {}

    for idx in data.index:
        mol = Chem.MolFromSmiles(data.loc[idx, "SMILES"])
        ECFP_bitinfo = {}
        ECFP = AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=ECFP_bitinfo)
        for f in ECFP_bitinfo.keys():
            if f not in new_columns_data:
                new_columns_data[f] = [0] * len(data)
            new_columns_data[f][idx] = len(ECFP_bitinfo[f])

    new_columns_df = pd.DataFrame(new_columns_data)
    new_columns_df.columns = new_columns_df.columns.astype(
        str
    )  ## converting column names to string type

    data = pd.concat([data, new_columns_df], axis=1)
    data = data.fillna(0)  ## fill NaN with 0
    data = data.loc[:, (data != data.iloc[0]).any()]
    data.drop(columns=["SMILES"], inplace=True)

    return data


def data_split(df, label="label", test_size=0.2):
    """
    Divide the dataset into a training set and a test set using stratified sampling.

    Parameters:
    -----------
    - df (pandas.DataFrame): The input DataFrame containing the dataset.
    - label (str): The name of the column that contains the target labels. Default is "label".
    - test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.

    Returns:
    --------
    - X_train (pandas.DataFrame): The feature matrix of the training set.
    - X_test (pandas.DataFrame): The feature matrix of the test set.
    - y_train (pandas.Series): The target labels of the training set.
    - y_test (pandas.Series): The target labels of the test set.

    Notes:
    ------
    - The function uses stratified sampling to ensure that the distribution of the target labels is preserved in both the training and test sets.
    - The `random_state` parameter is set to 0 to ensure reproducibility of the results.
    - The function returns the feature matrices and target labels for both the training and test sets.
    """

    y = df[label]
    X = df.drop(label, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=True, random_state=0
    )

    return X_train, X_test, y_train, y_test


def feature_selection(model, num_features, X_train, y_train, X_test):
    """
    Perform feature selection based on an embedding method.

    Parameters:
    -----------
    - model: The machine learning model used for feature selection. This should be a model that provides feature importances or coefficients.
    - num_features (int): The maximum number of features to select.
    - X_train (pandas.DataFrame): The feature matrix of the training set.
    - y_train (pandas.Series): The target labels of the training set.
    - X_test (pandas.DataFrame): The feature matrix of the test set.

    Returns:
    --------
    - X_train (pandas.DataFrame): The feature matrix of the training set after feature selection.
    - X_test (pandas.DataFrame): The feature matrix of the test set after feature selection.
    - features_selected (pandas.DataFrame): A DataFrame containing the names of the selected features.

    Notes:
    ------
    - The function uses the `SelectFromModel` class from `sklearn.feature_selection` to perform feature selection.
    - The `max_features` parameter is used to specify the maximum number of features to select.
    - The selected features are transformed back into DataFrames with the appropriate column names.
    - The function returns the transformed feature matrices for both the training and test sets, along with a DataFrame containing the names of the selected features.
    """

    selector = SelectFromModel(model, max_features=num_features).fit(X_train, y_train)
    feature_names = selector.get_feature_names_out().tolist()

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    features_selected = pd.DataFrame(feature_names, columns=["Features"])

    return X_train, X_test, features_selected


def calculate_corr_pval(df):
    """
    Calculate the correlation matrix and p-value matrix for a DataFrame.

    Parameters:
    -----------
    - df (DataFrame): The input DataFrame containing the data for which to calculate the correlation and p-values.

    Returns:
    --------
    Tuple: Returns a tuple containing two DataFrame objects, the first is the correlation matrix rounded to two decimal places,
    and the second is the p-value matrix.
    """

    corr_matrix = df.corr(method="spearman").round(2)

    pval_matrix = pd.DataFrame(
        np.zeros_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns
    )

    for i, col_i in enumerate(df.columns):
        for j, col_j in enumerate(df.columns):
            if i != j:
                _, p_val = spearmanr(df[col_i], df[col_j])
                pval_matrix.loc[col_i, col_j] = p_val

    return corr_matrix, pval_matrix


def evaluate_model(
    model, X_train, y_train, X_test, y_test, kf=5, name="", metric="accuracy"
):
    """
    Evaluate the performance of a binary classifier on training and test sets.

    Parameters:
    -----------
    - model: The machine learning model to evaluate.
    - X_train (pandas.DataFrame): The feature matrix of the training set.
    - y_train (pandas.Series): The target labels of the training set.
    - X_test (pandas.DataFrame): The feature matrix of the test set.
    - y_test (pandas.Series): The target labels of the test set.
    - kf (int): The number of folds for cross-validation. Default is 5.
    - name (str): The name of the model for display purposes. Default is an empty string.
    - metric (str): The evaluation metric to use for cross-validation. Default is "accuracy".

    Returns:
    --------
    - None: The function prints the evaluation results and displays a confusion matrix heatmap.

    Notes:
    ------
    - The function performs k-fold cross-validation on the training set and evaluates the model's performance.
    - It fits the model to the training set and makes predictions on the test set.
    - It prints the cross-validation results, Matthews Correlation Coefficient, and a classification report.
    - It also displays a confusion matrix heatmap for visualizing the performance on the test set.
    """

    val_acc = cross_val_score(model, X_train, y_train, cv=kf, scoring=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    mc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = ["Non-Toxic", "Toxic"]
    conf_matrix = pd.DataFrame(cm, index=classes, columns=classes)

    print(
        f"{name}'s results of 10-fold cross-validation are as follows: \n {val_acc} \n"
    )
    print(f"{name}'s mean result of 10-fold cross-validation is {val_acc.mean():.3g}")
    print(f"{name}'s Matthews Correlation Coefficient is {mc:.3g} \n")
    print(f"{name}'s performance on test set is as follows:\n{report}")

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 10}, cmap="Blues")
    plt.title(f"{name}", fontsize=15)
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.show()


def evaluate_model_3class(
    model, X_train, y_train, X_test, y_test, kf=5, name="", metric="f1_macro"
):
    """
    Evaluate the performance of a three-class classifier on training and test sets.

    Parameters:
    -----------
    - model: The machine learning model to evaluate.
    - X_train (pandas.DataFrame): The feature matrix of the training set.
    - y_train (pandas.Series): The target labels of the training set.
    - X_test (pandas.DataFrame): The feature matrix of the test set.
    - y_test (pandas.Series): The target labels of the test set.
    - kf (int): The number of folds for cross-validation. Default is 5.
    - name (str): The name of the model for display purposes. Default is an empty string.
    - metric (str): The evaluation metric to use for cross-validation. Default is "f1_macro".

    Returns:
    --------
    - None: The function prints the evaluation results and displays a confusion matrix heatmap.

    Notes:
    ------
    - The function performs k-fold cross-validation on the training set and evaluates the model's performance.
    - It fits the model to the training set and makes predictions on the test set.
    - It prints the cross-validation results, Matthews Correlation Coefficient, and a classification report.
    - It also displays a confusion matrix heatmap for visualizing the performance on the test set.
    - The target labels are assumed to have three classes: "Non-Toxic", "Low-Toxic", and "High-Toxic".
    """

    val_acc = cross_val_score(model, X_train, y_train, cv=kf, scoring=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    reports = classification_report(y_test, y_pred)
    mc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = ["Non-Toxic", "Low-Toxic", "High-Toxic"]
    conf_matrix = pd.DataFrame(cm, index=classes, columns=classes)

    print(
        f"{name}'s results of 10-fold cross-validation are as follows: \n {val_acc} \n"
    )
    print(f"{name}'s mean result of 10-fold cross-validation is {val_acc.mean():.3g}")
    print(f"{name}'s Matthews Correlation Coefficient is {mc:.3g} \n")
    print(f"{name}'s performance on test set is as follows:\n{reports}")

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 10}, cmap="Blues")
    plt.title(f"{name}", fontsize=15)
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.show()


def calculate_train_dt(train_set, k_neighbors=3, Z=0.5):
    """
    Calculate the Euclidean distance between a sample and its nearest neighboring samples.

    Parameters:
    -----------
    - train_set (numpy.ndarray): The training set where each row represents a sample.
    - k_neighbors (int): The number of nearest neighbors to consider. Default is 3.
    - Z (float): The coefficient used to adjust the standard deviation. Default is 0.5.

    Returns:
    --------
    - dt_values (list): A list of calculated DT (Distance Threshold) values for each sample in the training set.

    Notes:
    ------
    - For each sample in the training set, the function calculates the Euclidean distances to all other samples.
    - It then selects the k-nearest neighbors based on these distances.
    - The mean distance (`gamma_bar`) and standard deviation (`sigma`) of the nearest neighbors are computed.
    - The DT value for each sample is calculated as `gamma_bar + Z * sigma`.
    - The function returns a list of DT values for all samples in the training set.
    """

    dt_values = []

    for i, train_sample in enumerate(train_set):
        other_samples = np.delete(train_set, i, axis=0)
        distances = [
            euclidean(train_sample, other_sample) for other_sample in other_samples
        ]
        sorted_indices = np.argsort(distances)[:k_neighbors]
        nearest_distances = np.array([distances[i] for i in sorted_indices])

        gamma_bar = np.mean(nearest_distances)
        sigma = np.std(distances)
        dt = gamma_bar + Z * sigma

        dt_values.append(dt)

    return dt_values


def calculate_test_dt(test_set, train_set, k_neighbors=3, Z=0.5):
    """
    Calculate the mean Euclidean distance between each test sample and its nearest neighbors in the training set.

    Parameters:
    -----------
    - test_set (numpy.ndarray): A numpy array of test samples.
    - train_set (numpy.ndarray): A numpy array of training samples.
    - k_neighbors (int): The number of nearest neighbors to consider. Default is 3.
    - Z (float): The coefficient used to adjust the standard deviation. Default is 0.5.

    Returns:
    --------
    - dt_values (list): A list of calculated DT (Distance Threshold) values for each test sample.

    Notes:
    - For each test sample, the function calculates the Euclidean distances to all training samples.
    - It then selects the k-nearest neighbors based on these distances.
    - The mean distance (`gamma_bar`) and standard deviation (`sigma`) of the nearest neighbors are computed.
    - The DT value for each test sample is calculated as `gamma_bar + Z * sigma`.
    - The function returns a list of DT values for all test samples.
    """

    dt_values = []

    for test_sample in test_set:
        distances = [euclidean(test_sample, train_sample) for train_sample in train_set]
        sorted_indices = np.argsort(distances)[:k_neighbors]
        nearest_distances = [distances[i] for i in sorted_indices]

        gamma_bar = np.mean(nearest_distances)
        sigma = np.std(distances)
        dt = gamma_bar + Z * sigma
        dt_values.append(dt)

    return dt_values
