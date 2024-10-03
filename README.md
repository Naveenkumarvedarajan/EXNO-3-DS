# Ex:3 Feature Encoding And Transformation

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding

An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
 
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
 
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding

We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
  
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
  
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
## FEATURE ENCODING:
### 1.Ordinal Encoding
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Blue','Green','Red']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["nom_0"]])
df['bo2']=e1.fit_transform(df[["nom_0"]])
df
```
![3-1](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/077d5518-ee17-43b4-85c6-fb6f454df195)

### 2.Label Encoding
```
le=LabelEncoder()
dfc=df.copy()
dfc['nom_0']=le.fit_transform(dfc['nom_0'])
dfc
```
![3-2](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/fc627ee4-c01e-45ec-b7ac-52b0b2699ca8)

### 3.Binary Encoding
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_1'])
df1=pd.concat([df,nd],axis=1)
df2=df.copy()
df1
```
![3-3](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/307186c2-73e4-4937-85b2-93eae41a9789)

### 4.One Hot Encoding
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=dfc.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['ord_2']]))
df2=pd.concat([df2,enc],axis=1)
pd.get_dummies(df2,columns=["ord_2"])
df2
```
![3-4](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/2ae779d7-e5cf-49ad-85fb-9ce9a067fd02)

## Feature Transformation
### Log Transformation
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df.skew()
np.log(df["Highly Negative Skew"])
```
![3-5](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/3a980465-66b6-4607-8b72-ea69f2ad0831)

### Reciprocal Transformation
```
np.reciprocal(df["Moderate Negative Skew"])
```
![3-6](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/215711d4-bebe-4edd-940b-6d794884ef9a)

### Square Root Transformation
```
np.sqrt(df["Highly Negative Skew"])
```
![3-7](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/7f9bc3fe-4221-44f3-a9ea-1df337d1e60b)

### Square Transformation
```
np.square(df["Highly Negative Skew"])
```
![3-8](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/d9abbd77-1abf-4763-bbae-31b4855a0350)

### Boxcox method
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![3-9](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/6b3b8394-27b2-4cea-83bf-ee1a1bafc608)

### Yeojohnson method
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![3-10](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/0d773817-811c-4ffd-b167-c5b321af0854)

### Quantile Transformation
```
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![3-11](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/1c8db9ed-1d9a-40e2-b982-7fa2924f54bf)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![3-12](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/60598a16-198f-4b99-b944-91f94c3dccbe)

```
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![3-13](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/4cfb06fe-94a8-40e1-b61b-768eb471e381)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![3-14](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/cb3fb59a-9861-4109-8eb0-2a3f9957e926)

```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![3-15](https://github.com/Divya110205/EXNO-3-DS/assets/119404855/38668cca-1d91-441d-a477-9ef5a4a67336)

# RESULT:
  Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.    

       
