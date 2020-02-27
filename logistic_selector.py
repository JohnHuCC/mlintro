import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.metrics import classification_report
import seaborn as sns

def feature_shrink_or_zoom(x_train,x_test):
    sc =  StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    return x_train_std, x_test_std

def logistic_reg_train(x_train_std, y_train):
    lr = LogisticRegression(solver='liblinear')
    lr.fit(x_train_std, y_train)
    return lr

def get_logistic_reg_predict_probability(x_test_std, lr):
    return lr.predict_proba(x_test_std)

def bayes_inference_train(x_train_std, y_train):
    bnb = BernoulliNB()
    bnb.fit(x_train_std, y_train)
    return bnb

def get_correct_rate(classfication, x_test_std, y_test):
    classfication.predict(x_test_std)
    error = 0
    num_of_data = 0
    for i, v in enumerate(classfication.predict(x_test_std)):
        num_of_data += 1
        if v != y_test.iloc[i,]:
            error += 1
    correct_rate = 1 - (error/num_of_data)
    return correct_rate

def cut_train_test_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

def get_GOT_GPT_ratio(GOT, GPT):
    GOT_GPT_ratio = GOT/GPT
    return GOT_GPT_ratio

def data_correlation(Selector,GOT_GPT_ratio):
    GOT_GPT_data_correlation = GOT_GPT_ratio.corr(Selector)
    return GOT_GPT_data_correlation

def run_data(x,y):
    x_train, x_test, y_train, y_test = cut_train_test_data(x, y)
    x_train_std, x_test_std = feature_shrink_or_zoom(x_train, x_test)
    lr = logistic_reg_train(x_train_std, y_train)
    lr_correct_rate = get_correct_rate(lr, x_test_std, y_test)
    bnb = bayes_inference_train(x_train_std, y_train)
    bnb_correct_rate = get_correct_rate(bnb, x_test_std, y_test)
    return lr, lr_correct_rate,bnb, bnb_correct_rate

def get_mean_squared_log_error(reg_kind,x, y):
    x_train, x_test, y_train, y_test = cut_train_test_data(x, y)
    y_pred = reg_kind.predict(x_test)
    mean_squared_log_error = metrics.mean_squared_log_error(y_test,y_pred)
    return mean_squared_log_error

def main():
    data = pd.read_csv('liver.csv')
    x, y = data.iloc[0:, 0:6], data.iloc[0:, 6]
    lr, lr_correct_rate, bnb, bnb_correct_rate = run_data(x,y)
    lr_msle = get_mean_squared_log_error(lr, x, y)
    bnb_msle = get_mean_squared_log_error(bnb, x, y)
    print('lr_correct_rate:', lr_correct_rate)
    print('bnb_correct_rate:', bnb_correct_rate)
    print ('lr_mean_squared_log_error:', ls_msle)
    print('bnb_mean_squared_log_error:', bnb_msle)


    GOT = data.iloc[0:, 3]
    GPT = data.iloc[0:, 2]
    GOT_GPT_ratio = get_GOT_GPT_ratio(GOT, GPT)
    GPT_GOT_ratio = get_GOT_GPT_ratio(GPT, GOT)
    data.iloc[0:, 3] = GOT_GPT_ratio
    data.iloc[0:, 2] = GPT_GOT_ratio
    x_changed = data.iloc[0:,0:6]
    x_changed_only_GOT_GPT_ratio = data.iloc[0:,[0, 1, 3, 4, 5]]
    x_changed_only_GPT_GOT_ratio = data.iloc[0:, [0, 1, 2, 4, 5]]
    lr_correct_rate_changed, bnb_correct_rate_changed = run_data(x_changed, y)
    print('lr_correct_rate_changed:', lr_correct_rate_changed)
    print('bnb_correct_rate_changed:', bnb_correct_rate_changed)
    lr_correct_rate_changed, bnb_correct_rate_changed = run_data(x_changed_only_GOT_GPT_ratio, y)
    print('mean square error:', get_mean_squared_log_error(reg_kind,x, y))

    print('lr_correct_rate_changed_GOT_GPT_ratio:', lr_correct_rate_changed)
    print('bnb_correct_rate_changed_GPT_GOT_ratio:', bnb_correct_rate_changed)
    lr_correct_rate_changed, bnb_correct_rate_changed = run_data(x_changed_only_GPT_GOT_ratio, y)
    print('lr_correct_rate_changed:', lr_correct_rate_changed)
    print('bnb_correct_rate_changed:', bnb_correct_rate_changed)
    GPT_GOT_ratio_corrwith_selector = data_correlation(y, GPT_GOT_ratio)
    GOT_GPT_ratio_corrwith_selector = data_correlation(y, GOT_GPT_ratio)
    GPT_corrwith_selector = data_correlation(y,GPT)
    GOT_corrwith_selector = data_correlation(y,GOT)
    print('GPT_GOT_ratio_corrwith_selector:', GPT_GOT_ratio_corrwith_selector)
    print('GOT_GPT_ratio_corrwith_selector:', GOT_GPT_ratio_corrwith_selector)
    print('GPT_corrwith_selector:',GPT_corrwith_selector)
    print('GOT_corrwith_selector:',GOT_corrwith_selector)


main()