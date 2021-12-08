import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=["age","workclass","fnlwgt","education",
		"education_num","marital-status","occupation","relationship","race","sex","capital-gains","capital-loss","hours-per-week","native-country","income"])

def multi_occ_plot(df, typemask:[int]):
	for i,x in enumerate(df.columns.values.tolist()):
		match typemask[i]:
			case 1:
				df[x].value_counts().sort_index().plot(kind='bar', xlabel=x, ylabel="occurences")
			case 2:
				df[x].value_counts().sort_index().plot(xlabel=x, ylabel="occurences")
			case 3:
				df[x].value_counts().sort_index().plot(kind='box', xlabel=x)
		plt.show()
		