import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handles data cleaning, normalization, and splitting."""

    def __init__(self, df1):
        self.df1 = df1.copy()
        self.features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    def replace_zeros_with_median(self):
        """Replace zero values in selected columns with the median."""
        for feature in self.features:
            median_value = self.df1[feature].replace(0, np.nan).median()
            self.df1[feature] = self.df1[feature].replace(0, median_value)

    def normalize_features(self):
        """Standardize feature values and ensure correct dtype."""
        scaler1 = StandardScaler()
        numeric_cols = self.df1.select_dtypes(include=[np.number]).columns[:-1]
        self.df1[numeric_cols] = scaler1.fit_transform(self.df1[numeric_cols]).astype(np.float64)
        return scaler1  # Return scaler for future transformations

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X = self.df1.iloc[:, :-1].values.astype(np.float32)  # Ensure dtype consistency
        y = self.df1.iloc[:, -1].values.astype(np.float32)
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


class DiabetesDataset(Dataset):
    """Custom PyTorch Dataset for structured data handling."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape to column vector

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Visualization function
def plot_correlation_matrix(df1):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df1.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()


def plot_boxplots(df1):
    plt.figure(figsize=(12, 6))
    df1.boxplot(rot=45)
    plt.title("Boxplots of Features (Outlier Detection)")
    plt.show()


def plot_pairplot(df1):
    sns.pairplot(df1, hue="Outcome", diag_kind="kde")
    plt.suptitle("Pairplot of Features by Outcome", fontsize=16)
    plt.show()


def plot_class_distribution(df1):
    sns.countplot(x=df1["Outcome"])
    plt.title("Distribution of Outcome Classes")
    plt.show()


def plot_scaled_heatmap(df1):
    scaler1 = StandardScaler()
    scaled_data = scaler1.fit_transform(df1.drop(columns=["Outcome"]))
    scaled_df = pd.DataFrame(scaled_data, columns=df1.columns[:-1])

    plt.figure(figsize=(10, 6))
    sns.heatmap(scaled_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap (Scaled Data)")
    plt.show()


def plot_feature_distributions(df1):
    df1.hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.show()

# Example usage (for testing purposes)
if __name__ == "__main__":
    names = ["data_test.csv", "datalocal1.csv", "datalocal2.csv"]
    for name in names:

        df = pd.read_csv(name)
        preprocessor = DataPreprocessor(df)
        preprocessor.replace_zeros_with_median()
        scaler = preprocessor.normalize_features()
        X_train, X_test, y_train, y_test = preprocessor.split_data()

        train_dataset = DiabetesDataset(X_train, y_train)
        test_dataset = DiabetesDataset(X_test, y_test)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        sample_X, sample_y = next(iter(train_loader))
        print(sample_X.shape, sample_y.shape)

        plot_correlation_matrix(df)
        plot_class_distribution(df)
        

"""
    datalocal1= pd.read_csv("datalocal1.csv")
    datalocal2= pd.read_csv("datalocal2.csv")
    data_test= pd.read_csv("data_test.csv")

    # Show distributions to confirm difference
    distributions = {
        'datalocal1': datalocal1['Outcome'].value_counts(normalize=True),
        'datalocal2': datalocal2['Outcome'].value_counts(normalize=True),
        'data_test': data_test['Outcome'].value_counts(normalize=True)
    }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    pd.DataFrame(distributions).plot(kind='bar', ax=ax)
    ax.set_title('Different Outcome Distributions Across Splits')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Outcome')
    plt.tight_layout()
    plt.show()
"""
