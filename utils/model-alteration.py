import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier

class ModelAlteration():
    def strat_kfold_evaluation(
		self,
		df, 
		model, 
		target:int, 
		folds:int, 
		shuffle:bool=True, 
		random_state:int=None) -> [float, ([],[])]:
        '''
		Implements some centroid based clustering algorithms on n-dimensional data
			
		Parameters
		------------
		df			: Your dataframe
		model		: A scikitlearn model used to classify labels
		target 		: The index of your target column
		folds		: How often your dataframe should be split
		shuffle		: Specifies if the samples should be shuffled
		random_state: If shuffle=True, random_state specifies the used seed.
					if None, shuffle will always be random.
						
		Returns
		------------
		accuracy	: A list which contains the accuracy of the model over each folds
		best_fold	: The fold with the highest accuracy with the used model
		'''
	
        data, target = df.loc[:, df.columns!=target].values, df[target].values      
        skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        accuracy = [0 for _ in range(folds)]
        best_fold = []
        for i, index in enumerate(skf.split(data, target)):
            x_train, x_test = data[index[0]], data[index[1]]
            y_train, y_test = target[index[0]], target[index[1]]
            model.fit(x_train, y_train)
            accuracy[i] = (model.score(x_test, y_test))*100
            if accuracy[i] >= max(accuracy[:-1]): best_fold = index
        return(accuracy, best_fold)

    def plot_accuracy(self, acc:[[float]], xlab:str, legend:[str], xaxis:[]=[]):
        print(legend, xaxis)
        plt.xlabel(xlab)
        plt.ylabel('Accuracy [%]')
        acc = acc if len(acc)>0 else [acc]
        if not xaxis:
            for i, accuracy in enumerate(acc):
                plt.plot(range(len(accuracy)), accuracy, label = legend[i])
        else:
            for i, accuracy in enumerate(acc):
                plt.plot(xaxis, accuracy, label = legend[i])  
        plt.legend(loc="upper left")

    def optimize_knn(self, 
        df, 
        target:int,
        neighbours:[int] = list(range(1,11)), 
        metric:[int]=[1,2,3],
        folds:int = 10,
        plot:bool=True):
        '''
        Attempts to find the most optimal model parameters for the k-nearest-
        neighbours (kNN) classifier by finding the best fold for each permutation 
        of the parameters. The best fold is determined by strat_kfold_evaluation(). 
        The accuracy of all best folds is then compared and the  parameters of 
        the best fold are returned (in addition to the fold itself)

        Parameters
        ------------
        df : dataframe
            Your datatable
        target : int 
            The index of your target column
        neighbours : [int]
            A list which contains the number of neighbors which should be used in kNN.
        metric : [int] 
            Which metric should be used for kNN
            1 - Manhattan
            2 - Euclidean
            3>= - Minkowski
        folds : int 
            How often your dataframe should be split in strat_kfold_evaluation
        plot : bool
            Plots the accuracies over each fold

        Returns
        ------------
        best_fold: (np.array(int), np.array(int))
            An indexlist of the fold which has performed best overall
        k : int
            The number of neighbours where the model has performed best 
        used_metric : int 
            The metric used where the model performed best
        '''
        best_acc, best_model, fold_acc = 0, 0, [[None for _ in neighbours] for _ in metric]
        epoch, end = 1, len(neighbours)*len(metric)
        for i,m in enumerate(metric):
            for j,k in enumerate(neighbours):
                model = KNeighborsClassifier(n_neighbors=k, p = m)
                fold_acc[i][j], tmp_fold = (lambda x: [max(x[0]), x[1]])(self.strat_kfold_evaluation(df, model, target, folds))
                if fold_acc[i][j] > best_acc: 
                    best_acc = fold_acc[i][j]
                    best_model = (tmp_fold, k, m)
                print("Epoch %s/%s | neighbours=%s, metric=%s, Accuracy=%s" % (epoch, end, k, m, fold_acc[i][j]))
                epoch += 1
        if plot: self.plot_accuracy(fold_acc, "Number of neighbours", list(map(lambda x: "Metric " + x, list(map(str, metric)))), neighbours)
        return(best_model)

    def optimize_perceptron(self, 
        df, 
        target:int,
        learning_rate:[float] = np.linspace(1, 20, num=20), 
        penalty:[int]=[0,1,2,3],
        folds:int = 10,
        plot:bool=True):
        '''
        Attempts to find the most optimal model parameters for the perceptron 
        classifier by finding the best fold for each permutation of the 
        parameters. The best fold is determined by strat_kfold_evaluation(). 
        The accuracy of all best folds is then compared and the parameters of 
        the best fold are returned (in addition to the fold itself)

        Parameters
        ------------
        df : dataframe
            Your datatable
        target : int 
            The index of your target column
        learning_rate : [float]
            A list containing the number of learning_rates the algorithm should
            try out
        penalty : [int] 
            Which penalty should be used
            0 - None
            1 - l1
            2 - l2
            3 - elasticnet
        folds : int 
            How often your dataframe should be split in strat_kfold_evaluation
        plot : bool
            Plots the accuracies over each fold

        Returns
        ------------
        best_fold: (np.array(int), np.array(int))
            An indexlist of the fold which has performed best overall
        l : float
            The best learning rate for this dataset+penalty 
        r : int 
            Best penalty   
        '''
        best_acc, best_model, fold_acc = 0, 0, [[None for _ in learning_rate] for _ in penalty]
        epoch, end = 1, len(learning_rate)*len(penalty)
        penalty = list(map((lambda x, d={0:None, 1:"l1", 2:"l2", 3:"elasticnet"}: d[x]), penalty)) 

        for i, m in enumerate(penalty):
            for j, k in enumerate(learning_rate):
                model = Perceptron(eta0=k, penalty=m)
                fold_acc[i][j], tmp_fold = (lambda x: [max(x[0]), x[1]])(self.strat_kfold_evaluation(df, model, target, folds))
                if fold_acc[i][j] > best_acc: 
                    best_acc = fold_acc[i][j]
                    best_model = (tmp_fold, k, m)
                print("Epoch %s/%s | learning_rate=%s, penalty=%s, Accuracy=%s" % (epoch, end, k, m, fold_acc[i][j]))
                epoch += 1
        if plot: self.plot_accuracy(fold_acc, "Used learning_rate", list(map(lambda x: "penalty: " + str(x), penalty)), list(learning_rate))
        return(best_model)