# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/nitinsawhney/tourism-pkg-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset excluding target
numeric_features = [
    'Age',                      # Age of the customer
    'CityTier',                 # Tier of the city (Tier 1 > Tier 2 > Tier 3)
    'DurationOfPitch',          # Duration of the sales pitch delivered to the customer
    'NumberOfPersonVisiting',   # Total number of people accompanying the customer on the trip
    'NumberOfFollowups',        # Total number of follow-ups by the salesperson after the sales pitch
    'PreferredPropertyStar',    # Preferred hotel rating by the customer
    'NumberOfTrips',            # Average number of trips the customer takes annually
    'Passport',                 # Whether the customer holds a valid passport (0: No, 1: Yes)
    'PitchSatisfactionScore',   # Score indicating the customer's satisfaction with the sales pitch.
    'OwnCar',                   # Whether the customer owns a car (0: No, 1: Yes)
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer
    'MonthlyIncome'             # Gross monthly income of the customer
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # Country where the customer resides
    'Occupation',            # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender',                # Gender of the customer (Male, Female)
    'MaritalStatus',         # Marital status of the customer (Single, Married, Divorced)
    'ProductPitched',        # The type of product pitched to the customer
    'Designation'            # Customer's designation in their current organization    
]

## Performing Data preprocessing steps to ensure data quality

## Drop unwanted columns
# Let's drop the unnamed index column if it exists
if 'Unnamed: 0' in df.columns or df.columns[0] == '':
    df = df.iloc[:, 1:]

# Let's drop CustomerID as it's a unique identifier (not required for modeling)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

## Handle missing values, if any

# Let's handle missing values
print("\nIdentifying and handling missing values...")

# For numerical columns, fill with median
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, fill with mode
for col in categorical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

## Handle data inconsistencies in Gender column
# Let's handle specific data quality issues (for e.g., "Fe Male" should be "Female")

    df['Gender'] = df['Gender'].str.strip().replace(
        {'Fe Male': 'Female', 
          'Fe male': 'Female',
          'FE MALE': 'Female',
          'fe male': 'Female'
        })

# Let's standardize capitalization for future data handling

    df["Gender"] = df["Gender"].replace({
        "female": "Female",
        "male": "Male",
        "MALE": "Male",
        "FEMALE": "Female"
    })

print("\nData quality issues in Gender column handled.")

## Encoding Categorical columns based on the type of data
print("\nEncoding categorical variables...")

# Let's Encode Gender uding label encoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'].astype(str))

# Let's do one hot encoding for the rest of the categorical columns.
one_hot_columns = [
    'TypeofContact',
    'Occupation',
    'MaritalStatus',
    'ProductPitched',
    'Designation'
]

df = pd.get_dummies(
    df,
    columns=one_hot_columns,
    drop_first=True,
    dtype=int
)


# Split into X (features) and y (target)
X = df.drop(columns=[target])
y = df[target]

print(f"\nFeatures shape: {X.shape}")
print(f"\nTarget shape: {y.shape}")
print(f"\nTarget distribution:\n{y.value_counts()}")

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {Xtrain.shape[0]}")
print(f"Test set size: {Xtest.shape[0]}")

# Save the datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("\nDatasets saved locally.")

# Upload to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1], # just the filename
        repo_id="nitinsawhney/tourism-pkg-prediction",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face")

print("\nData preparation completed successfully!")
