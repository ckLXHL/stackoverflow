import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import label_binarize, MinMaxScaler
import seaborn as sns

def clean_data(df):
    '''
    INPUT
    df - pandas dataframe

    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector

    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns
    5. For each numeric variable, fill the column with the mean value.
    6. Create dummy columns for all the categorical variables, drop the original columns
    '''
    # Drop rows with missing salary values
    df = df.dropna(subset=['ConvertedSalary'], axis=0)
    y = df['ConvertedSalary']

    # Convert level data
    fe_map={
        'I never completed any formal education':0,
        'Primary/elementary school':1,
        'Some college/university study without earning a degree':2,
        'Associate degree': 3,
        'Bachelor’s degree (BA, BS, B.Eng., etc.)':4,
        'Master’s degree (MA, MS, M.Eng., MBA, etc.)':5,
        'Professional degree (JD, MD, etc.)':6,
        'Other doctoral degree (Ph.D, Ed.D., etc.)':7,
    }
    df.loc[:,'FormalEducation'] = df['FormalEducation'].map(fe_map)

    cs_map = {
        'Fewer than 10 employees':0,
        '10 to 19 employees':1,
        '20 to 99 employees':2,
        '100 to 499 employees':3,
        '500 to 999 employees':4,
        '1,000 to 4,999 employees':5,
        '5,000 to 9,999 employees':6,
        '10,000 or more employees':7
    }
    df.loc[:,'CompanySize'] = df['CompanySize'].map(cs_map)

    js_map = {
        'Extremely dissatisfied':0,
        'Moderately dissatisfied':1,
        'Slightly dissatisfied':2,
        'Neither satisfied nor dissatisfied':3,
        'Slightly satisfied':4,
        'Moderately satisfied':5,
        'Extremely satisfied':6
    }
    df.loc[:,'JobSatisfaction'] = df['JobSatisfaction'].map(js_map)

    hc_map = {
        'Less than 1 hour':0,
        '1 - 4 hours':1,
        '5 - 8 hours':2,
        '9 - 12 hours':3,
        'Over 12 hours':4
    }
    df.loc[:,'HoursComputer'] = df['HoursComputer'].map(hc_map)

    ho_map = {
    'Less than 30 minutes':0,
    '30 - 59 minutes':1,
    '1 - 2 hours':2,
    '3 - 4 hours':3,
    'Over 4 hours':4,
    }
    df.loc[:,'HoursOutside'] = df['HoursOutside'].map(ho_map)

    sm_map = {
        'Never':0,
        '1 - 2 times per week':1,
        '3 - 4 times per week':2,
        'Daily or almost every day':3
    }
    df.loc[:,'SkipMeals'] = df['SkipMeals'].map(sm_map)

    exer_map={
        "I don't typically exercise":0,
        '1 - 2 times per week': 1,
        '3 - 4 times per week':2,
        'Daily or almost every day':3
    }
    df.loc[:,'Exercise'] = df ['Exercise'].map(exer_map)

    yc_map = {
        '0-2 years':0,
        '3-5 years':1,
        '6-8 years':2,
        '9-11 years':3,
        '12-14 years':4,
        '15-17 years':5,
        '18-20 years': 6,
        '21-23 years': 7,
        '24-26 years':8,
        '27-29 years':9,
        '30 or more years':10
    }
    df.loc[:,'YearsCoding'] = df ['YearsCoding'].map(yc_map)
    df.loc[:,'CareerSatisfaction'] = df['CareerSatisfaction'].map(js_map)
    df.loc[:,'YearsCodingProf'] = df['YearsCodingProf'].map(yc_map)
    df.loc[:,'OpenSource'] = label_binarize(df['OpenSource'], classes =['Yes', 'No'])
    s = df['DevType'].str.split(';')
    s_d = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    for col in s_d.columns:
        df.loc[:,col] = s_d[col]

    #Drop respondent and expected salary columns
    df = df.drop(['Respondent', 'Salary', 'ConvertedSalary', 'DevType'], axis=1)


    # Fill numeric columns with the mean
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)

    # Dummy the categorical variables
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    df = pd.get_dummies(df, columns=cat_vars)
    
    # normolize
    #scaler = MinMaxScaler()
    #df.loc[:,:] = scaler.fit_transform(df.values) 

    return df, y

def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test

def main():
    df = pd.read_csv('survey_results_public.csv', low_memory=False)
    X, y = clean_data(df)
    #cutoffs here pertains to the number of missing values allowed in the used columns.
    #Therefore, lower values for the cutoff provides more predictors in the model.
    cutoffs = [5000, 3500, 2500, 1000, 100, 50, 30, 20, 10, 5]

    r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test = find_optimal_lm_mod(X, y, cutoffs, plot=False)
    print('Finished Finding the Best Model', r2_scores_test, r2_scores_train)
    return lm_model


if __name__ == '__main__':
    best_model = main()
