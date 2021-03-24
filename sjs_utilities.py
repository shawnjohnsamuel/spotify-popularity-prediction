import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, f1_score, roc_auc_score, plot_confusion_matrix
import eli5


def crossval(estimator, X, y, cv=5, scoring='precision'):
    '''
    Cross Fold Score with a default of 5 folds and score set to precision
    '''
    cv_scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    print(f"Avg {scoring.capitalize()} Score of {cv_scores.mean():.4f} with Std Dev of {cv_scores.std():.4f}")
    print('')
    print(f"The scores were: {list(map('{:.4f}'.format,cv_scores))}")

# Function designed for Flatiron Phase 3 Project - primarily for linear regression

# create dataframe to document scores
eval_metrics = pd.DataFrame(columns = ['Model', 'Precision', 'F1 Score', 'ROC-AUC'])

def evaluate(name, estimator, X_train, X_test, y_train, y_test, use_decision_function='yes'):
    '''
    Evaluation function to show a few scores for both the train and test set
    Also shows a confusion matrix for the test set
    
    use_decision_function allows you to toggle whether you use decision_function or
    predict_proba in order to get the output needed for roc_auc_score
    If use_decision_function == 'skip', then it ignores calculating the roc_auc_score
    
    *Created for Spotify Project*
    '''
    
    # grab predictions
    train_preds = estimator.predict(X_train)
    test_preds = estimator.predict(X_test)
    
    # output needed for roc_auc_score
    if use_decision_function == 'skip': # skips calculating the roc_auc_score
        train_out = False
        test_out = False
    elif use_decision_function == 'yes': # not all classifiers have decision_function
        train_out = estimator.decision_function(X_train)
        test_out = estimator.decision_function(X_test)
    elif use_decision_function == 'no':
        train_out = estimator.predict_proba(X_train)[:, 1] # proba for the 1 class
        test_out = estimator.predict_proba(X_test)[:, 1]
    else:
        raise Exception ("The value for use_decision_function should be 'skip', 'yes' or 'no'.")
      
    # plot test confusion matrix
    plot_confusion_matrix(estimator, X_test, y_test,
                          values_format=",.0f",
                          display_labels = ['Not Popular', 'Popular'])                            
    plt.title('Confusion Matrix (Test Set)')
    plt.show()
    
    # print scores
    print("          | Train  | Test   |")
    print("          |-----------------|")
    print(f"Precision | {precision_score(y_train, train_preds):.4f} | {precision_score(y_test, test_preds):.4f} |")
    print(f"F1 Score  | {f1_score(y_train, train_preds):.4f} | {f1_score(y_test, test_preds):.4f} |")
    if type(train_out) == np.ndarray:
        print(f"ROC-AUC   | {roc_auc_score(y_train, train_out):.4f} | {roc_auc_score(y_test, test_out):.4f} |")
    
    # add the row to metrics df and list in reverse order so current is at top
    new_row = []
    new_row.append(name)
    new_row.append(format(precision_score(y_test, test_preds),'.4f'))
    new_row.append(format(f1_score(y_test, test_preds),'.4f'))
    new_row.append(format(roc_auc_score(y_test, test_out),'.4f'))

    eval_metrics.loc[len(eval_metrics.index)] = new_row
    display(eval_metrics.sort_index(ascending=False, axis=0))


# Function designed for Flatiron Phase 2 Project - primarily for linear regression

linreg_metrics = pd.DataFrame(columns = ['Model Name', 'R2', 'MAE', 'RMSE']) #df to keep track of metrics for linereg_evaluate

def linreg_evaluate(name, model, df, continuous, categoricals, log=True, OHE=True, scale=True, scaler=MinMaxScaler(), seed=42, print=True):
    '''
    Performs a train-test split & evaluates a model
    
    Returns: model, scaler, y_train_pred, y_test_pred, X_test, y_test, X, y
   
    --
   
    Inputs:
     - name - string, name describing model
     - model - Instantiated sklearn model
     - df - pandas dataframe, containing all independent variables & target
     - continuous - list of all continuous independent variables
     - categoricals - list of all categorical independant variables
     - log - boolean, whether continuous variables should be logged 
     - OHE - boolean, whether categorical variables should be One Hot Encoded
     - scale - boolean, whether to scale the data with a MinMax Scaler
     - scaler - set to MinMaxScaler as default
     - seed - integer, for the random_state of the train test split

    Outputs (if print=True):
     - R2, MAE and RMSE for training and test sets
     - Scatter plot of risiduals from training and test sets
     - Stats model summary of model
     - Metrics Dataframe which lists R2, Mean Absolute Error & Root Mean Square. 
       (This df will be listed in reverse index, with latest results at the top)
        
    Returns:
     - model - fit sklearn model
     - scaler - fit scaler
     - y_train_preds - predictions for the training set
     - y_test_preds - predictions for the test set
     - X_test, y_test - if needed for OLS
     - X, y - if needed for final model
    '''
    
    preprocessed = df.copy()
    
    if log == True:
        pp_cont = preprocessed[continuous]
        log_names = [f'{column}_log' for column in pp_cont.columns]
        pp_log = np.log(pp_cont)    
        pp_log.columns = log_names
        preprocessed.drop(columns = continuous, inplace = True)
        preprocessed = pd.concat([preprocessed['price'], pp_log, preprocessed[categoricals]], axis = 1)
    else:
        preprocessed = pd.concat([preprocessed['price'], preprocessed[continuous], preprocessed[categoricals]], axis = 1)
        
    if OHE == True:
        preprocessed = pd.get_dummies(preprocessed, prefix = categoricals, columns = categoricals, drop_first=True)
 
    # define X and y       
    X_cols = [c for c in preprocessed.columns.to_list() if c not in ['price']]
    X = preprocessed[X_cols]
    y = preprocessed.price
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                        random_state=seed)
    
    # scale
    if scale == True:
        scaler = scaler
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    # fit model
    model.fit(X_train, y_train)

    # predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
        
    # calculate residuals
    residuals_train = y_train_pred - y_train
    residuals_test = y_test_pred - y_test
    
    if print == True:
        # print train and test R2, MAE, RMSE
        print(f"Train R2: {r2_score(y_train, y_train_pred):.3f}")
        print(f"Test R2: {r2_score(y_test, y_test_pred):.3f}")
        print("---")
        print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
        print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
        print("---")
        print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
        print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
                    
        # risduals plot training and test predictions
        plt.figure(figsize=(7,5))
        plt.scatter(y_train_pred, residuals_train, alpha=.75, label = "Train")
        plt.scatter(y_test_pred, residuals_test, color='g', alpha=.75, label = "Test")
        plt.axhline(y=0, color='black')
        plt.legend()
        plt.title(f'Residuals for {name}')
        plt.ylabel('Residuals')
        plt.xlabel('Predicted Values')
        plt.show()
    
        # display feature weights using ELI5
        display(eli5.show_weights(model, feature_names=X_cols))
        
        # add name, metrics and description to new row
        new_row = []
        new_row.append(name)
        new_row.append(format(r2_score(y_test, y_test_pred),'.3f'))
        new_row.append(format(mean_absolute_error(y_test, y_test_pred),'.2f'))
        new_row.append(format(np.sqrt(mean_squared_error(y_test, y_test_pred)),'.2f'))
    
        # add the row to metrics df and list in reverse order so current is at top
        linreg_metrics.loc[len(linreg_metrics.index)] = new_row
        display(linreg_metrics.sort_index(ascending=False, axis=0))

    return model, scaler, y_train_pred, y_test_pred, X_test, y_test, X, y