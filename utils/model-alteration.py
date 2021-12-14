import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB

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
        plt.show()

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
        best_fold: (np.array(int), {model_parameters})
	l 	 : An indexlist of the fold which has performed best overall
        dic	 : And a dict with the model parameters for the best fold 
        '''
        best_acc, best_model, fold_acc = 0, 0, [[None for _ in neighbours] for _ in metric]
        epoch, end = 1, len(neighbours)*len(metric)
        for i,m in enumerate(metric):
            for j,k in enumerate(neighbours):
                model = KNeighborsClassifier(n_neighbors=k, p = m)
                fold_acc[i][j], tmp_fold = (lambda x: [max(x[0]), x[1]])(self.strat_kfold_evaluation(df, model, target, folds))
                if fold_acc[i][j] > best_acc: 
                    best_acc = fold_acc[i][j]
                    best_model = (tmp_fold, {"n_neighbors" : k, "p" : m})
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
        best_fold: (np.array(int), {model_parameters})
	l 	 : An indexlist of the fold which has performed best overall
        dic	 : And a dict with the model parameters for the best fold 
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
                    best_model = (tmp_fold, { "eta0" : k, "penalty" : m})
                print("Epoch %s/%s | learning_rate=%s, penalty=%s, Accuracy=%s" % (epoch, end, k, m, fold_acc[i][j]))
                epoch += 1
        if plot: self.plot_accuracy(fold_acc, "Used learning_rate", list(map(lambda x: "penalty: " + str(x), penalty)), list(learning_rate))
        return(best_model)

    def optimize_SVM(self, 
        df, 
        target:int,
        regularization:[float] = np.linspace(1, 10, num=10), 
        kernel:[int]=[1,2,3],
        folds:int = 10,
        plot:bool=True):
        '''
        Attempts to find the most optimal model parameters for the SVM
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
        regularization: [float]
            A list containing all penalties which should be tried out on the 
            respective kernel function
        kernel : [int] 
            Which kernel functions should be used (refers to sklearn.svm.SVC)
            0 - Linear          (Takes a long time without dimension reduction)
            1 - Poly
            2 - rbf
            3 - sigmoid
            4 - precomputed     (Look at Sklearns documentary first if you want to use it)
        folds : int 
            How often your dataframe should be split in strat_kfold_evaluation
        plot : bool
            Plots the accuracies over each fold if True
        Returns
        ------------
        best_fold: (np.array(int), {model_parameters})
	l 	 : An indexlist of the fold which has performed best overall
        dic	 : And a dict with the model parameters for the best fold 
        '''
        best_acc, best_model, fold_acc = 0, 0, [[None for _ in regularization] for _ in kernel]
        epoch, end = 1, len(regularization)*len(kernel)
        kernel = list(map((lambda x, d={0:"linear", 1:"poly", 2:"rbf", 3:"sigmoid"}: d[x]), kernel)) 

        for i, kern in enumerate(kernel):
            for j, reg in enumerate(regularization):
                model = SVC(C=reg, kernel=kern)
                fold_acc[i][j], tmp_fold = (lambda x: [max(x[0]), x[1]])(self.strat_kfold_evaluation(df, model, target, folds))
                if fold_acc[i][j] > best_acc: 
                    best_acc = fold_acc[i][j]
                    best_model = (tmp_fold, {"C" :reg, "kernel" :kern})
                print("Epoch %s/%s | regularization = %s, kernel = %s, Accuracy = %s" % (epoch, end, reg, kern, fold_acc[i][j]))
                epoch += 1
        if plot: self.plot_accuracy(fold_acc, "Used regularization", list(map(lambda x: "kernel: " + str(x), kernel)), list(regularization))
        return(best_model)


    def optimize_decision_tree(self, 
        df, 
        target:int,
        criterion = ["gini", "entropy"], 
        max_depth:[int]= np.linspace(1, 10, num=10),
        splitter = ["best", "random"],
        folds:int = 10,
        plot:bool=True):
        '''
        Attempts to find the most optimal model parameters for the decision tree 
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
        criterion : [String]
            A list containing "gini" and "entropy"
        max_depth : [int] 
            A list containing the number of max_depth the algorithm should
            try out
        splitter : [String] 
            A list containing "best" and "random"
        folds : int 
            How often your dataframe should be split in strat_kfold_evaluation
        plot : bool
            Plots the accuracies over each fold
        Returns
        ------------
        best_fold: (np.array(int), {model_parameters})
	l 	 : An indexlist of the fold which has performed best overall
        dic	 : And a dict with the model parameters for the best fold 
        '''
        best_acc, best_model, fold_acc = 0, 0, [[[None for _ in max_depth] for _ in splitter] for _ in criterion]
        epoch, end = 1, len(criterion)*len(splitter)*len(max_depth)

        for i, cri in enumerate(criterion):
            for j, split in enumerate(splitter):
                for k, max_d in enumerate(max_depth):
                    model = DecisionTreeClassifier(criterion = cri, splitter = split, max_depth = max_d)
                    fold_acc[i][j][k], tmp_fold = (lambda x: [max(x[0]), x[1]])(self.strat_kfold_evaluation(df, model, target, folds))
                    if fold_acc[i][j][k] > best_acc: 
                        best_acc = fold_acc[i][j][k]
                        best_model = (tmp_fold, {"criterion": cri, "splitter": split, "max_depth": max_d})
                    print("Epoch %s/%s | criterion = %s, splitter = %s, max_depth = %s, Accuracy = %s" % (epoch, end, cri, split, max_d, fold_acc[i][j][k]))
                    epoch += 1
        for i in range(len(fold_acc)):
            if plot: self.plot_accuracy(fold_acc[i], "Used max depth", list(map(lambda x, y : "criterion and splitter: " + str(x) + " " + str(y), criterion,splitter)), list(max_depth))
        return(best_model)


    def optimize_NB(self, 
        df, 
        target:int,
        alpha:[float]= np.linspace(1, 10, num=10),
        fit_prior:[bool] = [True, False], 
        folds:int = 10,
        plot:bool=True):
        '''
        Attempts to find the most optimal model parameters for the NB 
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
        fit_prior : [bool]
            A list of True and False
        alpha : [int] 
            A list containing the number of alpha, the algorithm should
            try out
        folds : int 
            How often your dataframe should be split in strat_kfold_evaluation
        plot : bool
            Plots the accuracies over each fold
        Returns
        ------------
        best_fold: (np.array(int), {model_parameters})
	l 	 : An indexlist of the fold which has performed best overall
        dic	 : And a dict with the model parameters for the best fold 
        '''

        best_acc, best_model, fold_acc = 0, 0, [[None for _ in alpha ] for _ in fit_prior]
        epoch, end = 1, len(alpha)*len(fit_prior)

        for i, fit_p in enumerate(fit_prior):
            for j, alp in enumerate(alpha):
                model = ComplementNB(alpha = alp, fit_prior = fit_p)
                fold_acc[i][j], tmp_fold = (lambda x: [max(x[0]), x[1]])(self.strat_kfold_evaluation(df, model, target, folds))
                if fold_acc[i][j] > best_acc: 
                    best_acc = fold_acc[i][j]
                    best_model = (tmp_fold, {"alpha" : alp, "fit_prior" : fit_p})
                print("Epoch %s/%s | fit_prior = %s, alpha = %s, Accuracy = %s" % (epoch, end, fit_p, alp, fold_acc[i][j]))
                epoch += 1
        if plot: self.plot_accuracy(fold_acc, "Used alpha",  list(map(lambda x: "fit_prior: " + str(x), fit_prior)), list(alpha))
        return(best_model)
