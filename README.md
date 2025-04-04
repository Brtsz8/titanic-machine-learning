# 🚢 Titanic Survival Prediction

A machine learning project focused on predicting passenger survival from the Titanic disaster using various models, including Random Forest, KNN, Gaussian Naive Bayes, and Extra Trees. The project includes deep feature engineering, preprocessing pipelines, visualization, and hyperparameter tuning.

---

## 📌 Project Overview

Using the classic Titanic dataset, the goal is to explore the relationship between various passenger features and their likelihood of survival. This includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Building multiple ML pipelines
- GridSearchCV for hyperparameter tuning
- Model performance evaluation

---

## 📂 Dataset

The dataset is sourced from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data):

- `train.csv` – contains labeled training data
- `test.csv` – contains unlabeled test data for predictions

---

## 📊 Exploratory Data Analysis (EDA)

The EDA focused on understanding survival correlations and relationships between different features.
### Correlation Heatmap
```python
sns.heatmap(data_train.select_dtypes(include=["number"]).corr(), annot=True, cmap="coolwarm")
```

![TitanicDatasetHeatmap](https://github.com/user-attachments/assets/f8f65330-b8a4-4f45-86fc-fe625c46049b)

### Distribution Analysis

Age Distribution
```python
sns.displot(data_train, x='Age', col='Survived', binwidth=5, kde=True, color="red")
```
![Age](https://github.com/user-attachments/assets/f739d2bc-9687-49fd-a7bc-6d6b43239e5d)

Fare Distribution
```python
sns.displot(data_train, x='Fare', col='Survived', binwidth=25, kde=True, color="red")
```
![Fare](https://github.com/user-attachments/assets/55a69a30-e248-4870-9fe1-4faedb5b2946)

Name Length KDE
```python
sns.kdeplot(data=data_train , x='NameLength', hue='Survived', fill=True, palette="coolwarm")
```
![NameLength](https://github.com/user-attachments/assets/56ef85db-4c6a-4e01-8e02-e028759d0f6b)

## 🛠️ Feature Engineering
- `FamSize` = `SibSp` + `Parch` + 1

- `FamSizeGroup` (categorical grouping)

- `AgeGroup`, `FareGroup`, and `NameLengthGroup` using pd.qcut

- `OwnCabin` binary feature

- Binning Age, Fare, and NameLength manually for model interpretability
```python
bins = [0, 14, 19, 22, 25, 28, 31.8, 36, 41, 50, 80]
labels = list(range(10))
data_train['Age'] = pd.cut(data_train['Age'].fillna(meanAge), bins=bins, labels=labels).astype(int)
```

## 🤖 Model Training & Evaluation

### 1. Random Forest Classifier
```python
RandomForestClassifier(n_estimators=100, max_depth=10, ...)
Best Score: 83.23%
```
### 2. K-Nearest Neighbors
```python
KNeighborsClassifier(n_neighbors=11, algorithm='kd_tree', p=1)
Best Score: 80.09%
```
### 3. Gaussian Naive Bayes
```python
GaussianNB(var_smoothing=1e-08)
Best Score: 62.12%
```
(this is the model that did the worst)
### 4.Extra Trees Classifier
```python
ExtraTreesClassifier(n_estimators=300, min_samples_split=10, ...)
Score: (check output)
```
All models were trained using GridSearchCV with `StratifiedKFold(n_splits=5)` for reliable cross-validation.

## ✅ Best Performing Model
```
Model: Random Forest Classifier
Best Params: {'class_weight': None, 'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 100}
Best CV Accuracy: 83.23%
```
## 📌 Key Insights
- Having a cabin is strongly correlated with higher survival probability.

- Larger family size tends to negatively impact survival past a certain size.

- Name length has an unexpected but observable influence on survival rate.

