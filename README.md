import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("Titanic-Dataset.csv")     
print("Dataset Loaded Successfully")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = data[features]
y = data["Survived"]

imputer = SimpleImputer(strategy="median")
X.loc[:, "Age"] = imputer.fit_transform(X[["Age"]])

imputer2 = SimpleImputer(strategy="most_frequent")
X.loc[:, "Embarked"] = imputer2.fit_transform(X[["Embarked"]]).ravel()

label = LabelEncoder()
X["Sex"] = label.fit_transform(X["Sex"])
X["Embarked"] = label.fit_transform(X["Embarked"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

sex_encoder_for_sample = LabelEncoder()
sex_encoder_for_sample.fit(data["Sex"])

sample = pd.DataFrame({
    "Pclass": [3],
    "Sex": sex_encoder_for_sample.transform(["male"]),
    "Age": [22],
    "SibSp": [1],
    "Parch": [0],
    "Fare": [7.25],
    "Embarked": label.transform(["S"])
})

survival = model.predict(sample)[0]
print("Survived" if survival == 1 else "Not Survived")
