import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics


#iloc[列,行]

def set_print():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.width', 5000)

def data_fill_missing_value(data):
    data = data.fillna(data.mean())
    return data

def data_coding(data):
    data = pd.get_dummies(data)
    return data

def data_correlation_all():
    data_all = data.iloc[0:, 1:37]
    data_correlation_all = data_all.corr()
    return data_correlation_all

def draw_heatmap(data_correlation_all):
    sns.heatmap(data_correlation_all, annot=True)

def data_correlation():
    data_correlation = data.corrwith(data_SalePrice)
    return data_correlation

def find_the_most_corr(data_correlation):
    list1 = data_correlation.rank()
    for i in range(290):
        if (list1[i] >= 285.0):
            print(i)

def z_score_normalize(data):
    data_mean = data.mean()
    data_std_deviation = data.std()
    z_score_normalized = (data - data_mean) / data_std_deviation
    return z_score_normalized

def max_min_normalize(data):
    max_min_normalized = (data - data.min()) / (data.max() - data.min())
    return max_min_normalized

def data_normalize_func(x):
    x_std = preprocessing.scale(x)
    return x_std

def get_the_most_related_data(data_normalized):
    #x_the_most_related = z_score_normalized.iloc[0:, [4, 12, 16, 26, 27]]
    x_the_most_related = data_normalized.iloc[0:, [4, 12, 16, 26, 27]]
    return x_the_most_related

def get_the_least_related_data(data_normalized):
    #x_the_least_related = data.iloc[0:, [167, 172, 187, 229, 250]]
    x_the_least_related = data_normalized.iloc[0:, [167, 172, 187, 229, 250]]
    return x_the_least_related

def cut_train_test_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

def linear_regression_pred(x_train, y_train):
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    return linreg

def ridge_regression_pred(x_train, y_train):
    ridge_reg = Ridge(1)
    ridge_reg.fit(x_train, y_train)
    return ridge_reg

def lasso_regression_pred(x_train, y_train):
    lasso_reg = Lasso(1)
    lasso_reg.fit(x_train, y_train)
    return lasso_reg

def get_mean_squared_log_error(reg_kind,x_test, y_test):
    y_pred = reg_kind.predict(x_test)
    mean_squared_log_error = metrics.mean_squared_log_error(y_test,y_pred)
    return mean_squared_log_error

def draw_scatter_plot(data, x_test,y ,y_pred_ridge, y_test):
    y_real_col = data.iloc[:, 0]
    y_pred_col = x_test.iloc[:, 0]
    y_pred_col -= 1
    data_GrLivArea = data.iloc[0:, 16]
    data_SalePrice = data.iloc[0:, 37]
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.scatter(y_real_col, y, c='g')
    plt.scatter(y_pred_col, y_pred_ridge, c='b')
    plt.xlabel('the number of datas')
    plt.ylabel('value of house-prices')

    plt.subplot(412)
    plt.scatter(data_GrLivArea, data_SalePrice, c='g')
    plt.xlabel('the value of GrLivArea')
    plt.ylabel('value of house-prices')

    data = data_abnormal_clean(data)
    data_GrLivArea = data.iloc[0:, 16]
    data_SalePrice = data.iloc[0:, 37]
    plt.subplot(413)
    plt.scatter(data_GrLivArea, data_SalePrice, c='g')
    plt.xlabel('the value of GrLivArea(After clean outliners)')
    plt.ylabel('value of house-prices(After clean outliners)')

    plt.subplot(414)
    plt.scatter(y_test, y_pred_ridge, c='black')
    plt.xlabel('value of house-prices(real)')
    plt.ylabel('value of house-prices(predict)')
    plt.show()



def data_abnormal_clean(data):
    for i in range(data.index.max()):
        if any([data.loc[i,'GrLivArea']>3000,
                data.loc[i,'SalePrice']>400000]
                ):
            data.drop([i],inplace=True)
    return data

def linear_regression_pred_after_pred(data,linreg_std):
    x, y = data.iloc[0:, 0:37], data.iloc[0:, 37]
    x_std = preprocessing.scale(x)
    x_std_train, x_std_test, y_std_train, y_std_test = train_test_split(x_std, y, test_size=0.3, random_state=0)
    linreg_std.fit(x_std_train, y_std_train)
    y_std_pred_linear = linreg_std.predict(x_std_test)
    print("MSE_linear_std_after_clean:", metrics.mean_squared_log_error(y_std_test, y_std_pred_linear))

def main():
    set_print()
    data = pd.read_csv('house-prices.csv')
    data = data_fill_missing_value(data)
    data = data_coding(data)
    x, y = data.iloc[0:, 0:37], data.iloc[0:, 37]
    x_train,x_test,y_train,y_test = cut_train_test_data(x, y)
    z_score_normalized = z_score_normalize(data)
    x_z_score, y_z_score = z_score_normalized.iloc[0:, 0:37], z_score_normalized.iloc[0:, 37]
    x_z_score_train, x_z_score_test, y_z_score_train, y_z_score_test = cut_train_test_data(x_z_score, y)
    x_std = data_normalize_func(x)
    x_the_most_related = get_the_most_related_data(z_score_normalized)
    x_the_least_related = get_the_least_related_data(z_score_normalized)
    x_the_most_related_std = data_normalize_func(x_the_most_related)
    x_the_least_related_std = data_normalize_func(x_the_least_related)
    x_std_train, x_std_test, y_std_train, y_std_test = cut_train_test_data(x_std, y)
    x_the_most_related_std_train, x_the_most_related_std_test, y_the_most_related_std_train, y_the_most_related_std_test = cut_train_test_data(
        x_the_most_related_std, y)
    x_the_least_related_std_train, x_the_least_related_std_test, y_the_least_related_std_train, y_the_least_related_std_test = cut_train_test_data(
        x_the_least_related_std, y)
    linreg = linear_regression_pred(x_train,y_train)
    linreg_z_score = linear_regression_pred(x_z_score_train,y_z_score_train)
    linreg_std = linear_regression_pred(x_std_train,y_std_train)
    linreg_the_most_related_std = linear_regression_pred(x_the_most_related_std_train,y_the_most_related_std_train)
    linreg_the_least_related_std = linear_regression_pred(x_the_least_related_std_train,y_the_least_related_std_train)
    ridge_reg = ridge_regression_pred(x_train,y_train)
    lasso_reg = lasso_regression_pred(x_train,y_train)
    y_pred_linear = linreg.predict(x_test)
    y_z_score_pred_linear = linreg_z_score.predict(x_z_score_test)
    y_std_pred_linear = linreg_std.predict(x_std_test)
    y_the_most_related_std_pred_linear = linreg_the_most_related_std.predict(x_the_most_related_std_test)
    y_the_least_related_std_pred_linear = linreg_the_least_related_std.predict(x_the_least_related_std_test)
    y_pred_ridge = ridge_reg.predict(x_test)
    y_pred_lasso = lasso_reg.predict(x_test)
    print("MSE_linear:", metrics.mean_squared_error(y_test, y_pred_linear))
    print("MSE_linear_z_score:", metrics.mean_squared_error(y_z_score_test, y_z_score_pred_linear))
    print("MSE_linear_std:", metrics.mean_squared_error(y_std_test, y_std_pred_linear))
    print("MSE_linear_the_most_related_std:",
          metrics.mean_squared_error(y_the_most_related_std_test, y_the_most_related_std_pred_linear))
    print("MSE_linear_the_least_related_std:",
          metrics.mean_squared_error(y_the_least_related_std_test, y_the_least_related_std_pred_linear))
    print("MSE_ridge:",metrics.mean_squared_error(y_test,y_pred_ridge))
    print("MSE_lasso:",metrics.mean_squared_error(y_test,y_pred_lasso))
    draw_scatter_plot(data, x_test, y, y_pred_ridge, y_test)

main()