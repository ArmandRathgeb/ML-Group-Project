# ML Group Project

## Installation
By default the `environment.yml` uses the CPU versions of numpyro and torch. Install the CUDA versions manually if you want to use them.
```
conda env create --file environment.yml
conda activate cancerclass
```

## Using the Library
You can either modify the [sample.ipynb](./sample.ipynb) file and run your analysis manually, or you can use the library manually.

```py
import cancerclass as cacl
...
model = cacl.train_pipeline(X_train, Y_train)
cacl.analysis_pipeline(model, (X_train,X_test,Y_train,Y_test), labels)
```

The invdividual elements of our pipeline are also available as standalone components that implement scikit-learn's `BaseEstimator` API.
```py
import cancerclass as cacl
from sklearn.model_selection import train_test_split

imputer = cacl.DreamAIImputer()
imputer.fit(data)
imputed_data = imputer.transform(data)

X_train,X_test,Y_train,Y_test = train_test_split(imputed_data, labels, shuffle=True, train_size=0.8)

model = cacl.BayesLogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(Y_train)
print(cacl.categorical_accuracy(Y_true, Y_pred))
```

## Testing Dataset
Our [testing dataset](./data) was taken from [this dataset](https://www.kaggle.com/datasets/piotrgrabo/breastcancerproteomes/) on kaggle.
It contains protein expression data, about ~12,000 features taken from 83 patients. Cancer classifications exist for each patient which we use as labels for testing.