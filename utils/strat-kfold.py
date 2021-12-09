import pandas as pd
import numpy as np

def strat_kfold_evaluation(df, model, target:int, folds:int) -> [float, (np.array(),np.array())]:
	'''
	Implements some centroid based clustering algorithms on n-dimensional data
		
	Parameters
	------------
	df			: Your dataframe
	model		: A scikitlearn model used to classify labels
	target 		: The index of your target column
	folds		: How often your dataframe should be split
					  
	Returns
	------------
	accuracy	: A list which contains the accuracy of the model over each folds
	best_fold	: The fold with the highest accuracy with the used model
	'''
	
    data, target = df.loc[:, df.columns!=target].values, df[target].values      
    skf = StratifiedKFold(n_splits=folds)
    accuracy = [0 for _ in range(folds)]
    best_fold = []
    for i, index in enumerate(skf.split(data, target)):
        x_train, x_test = data[index[0]], data[index[1]]
        y_train, y_test = target[index[0]], target[index[1]]
        model.fit(x_train, y_train)
        accuracy[i] = (model.score(x_test, y_test))
        if accuracy[i] >= max(accuracy[:-1]): best_fold = index
    return(accuracy, best_fold)