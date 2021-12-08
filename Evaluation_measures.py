class Evaluation_of_Classifier():
    '''
        Attributes:
        y (np.array())              : This is where we store the y_test data

        y_predicted (np.array())    : This is where we store the predicted lables from the modle

        clf (object)                : This is where we store the modle

        X (np.array())              : This is where we store the X_test data
    '''
    def __init__ (self, y, y_predicted, clf, X):
        self.y = y
        self.y_predicted = y_predicted
        self.class_lables = np.unique(y)
        self.clf = clf
        self.X = X   

    def evaluate(self):
        '''
            Evaluates the modle via confusion matrix (+ accuracy, error_rate, sensitivity, 
                                                            specificity, precision, F1 and ROC curve)
        '''
        confusion_matrix    = self.multi_class_confusion_matrix() 
        accuracy, error_rate, sensitivity, specificity, precision, F1 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        for i in range(len(self.class_lables)):
            TP = confusion_matrix[i,i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i, :]) - TP
            TN = np.sum(confusion_matrix) - np.sum(confusion_matrix[:,i]) - np.sum(confusion_matrix[i,:]) + confusion_matrix[i,i]
                
            sensitivity = np.append(sensitivity,TP / (TP + FN) )
            specificity = np.append(specificity,TN / (TN + FP))

            F1 = np.append(F1, TP/(TP + (1/2)*(FP+FN)))
            if (TP + FP) == 0:
                precision = np.append(precision,0) 
            else:
                precision = np.append(precision,TP / (TP + FP)) 

        accuracy = np.round((np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)), 2)
        error_rate = 1 - accuracy

        print("The confusion matrix")
        print(pd.DataFrame(data=confusion_matrix, index=["Positive (P)", "Negative (N)"], columns=["Positive (PP)",	"Negative (PN)"]))
        
        print(pd.DataFrame(data=np.array([accuracy*100, error_rate*100]), 
                        index=["Accuracy in %", "Error_rate in %"] , columns = [" "] ))
        print("\n")
        print(pd.DataFrame(data=np.array([np.round(sensitivity*100,decimals=2), np.round(specificity*100, decimals = 2), 
                                          np.round(precision*100, decimals= 2), np.round(F1*100, decimals= 2)]), 
            index=["Sensitivity in %", "Specificity in %", "Precision in %", "F1 in %"], columns=["income <= 50K", "income > 50K"]))                       
        print("\n")

        self.Roc()

        print("\n")
            
    def multi_class_confusion_matrix(self):
        '''
            Calculats the confusion matrix

            Returns:
                confusion_matrix (np.array()) : Is the confusion matrix in the right shape
        '''
        confusion_matrix = np.array([])
        for alable in self.class_lables:
            for plable in self.class_lables:
                
                if alable == plable: 
                    confusion_matrix = np.append(confusion_matrix, np.sum([alable == yt == pred_y 
                                                                           for yt, pred_y in zip(self.y, self.y_predicted)]))
                else: 
                    confusion_matrix = np.append(confusion_matrix, np.sum([alable == yt and plable == pred_y 
                                                                           for yt, pred_y in zip(self.y, self.y_predicted)]))
        return confusion_matrix.reshape(len(self.class_lables), len(self.class_lables))

    def Roc(self):
        '''
            Calculats und plots the Roc curve
        '''
        probs = self.clf.predict_proba(self.X)
        preds = probs[:,1]
        fpr, tpr, threshold = metrics.roc_curve(self.y, preds)

        roc_auc_score(self.y, preds)
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


# Example:
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import roc_auc_score
# import sklearn.metrics as metrics

# DT_model = DecisionTreeClassifier()
# DT_model = DT_model.fit(X_train,y_train)
# y_pred = DT_model.predict(X_test)

# Evaluation = Evaluation_of_Classifier(y_test, y_pred, DT_model, X_test)
# Evaluation.evaluate()