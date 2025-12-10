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
```

## Testing Dataset
