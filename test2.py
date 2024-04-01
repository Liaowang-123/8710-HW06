import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_excel('crime2.xlsx', engine='openpyxl')
# 6a
# 假设你有一个名为'data'的DataFrame，包含各种暴力犯罪的数量和暴力犯罪率

# 计算暴力犯罪率
data['violent'] = data['murder'] + data['rape'] + data['robbery'] + data['assault']

# 计算自然对数的变化 ∆ ln violent = ln(violentt) - ln(violentt-1)
data['delta_ln_violent'] = np.log(data['violent']).diff()
data['percent_violent'] = data['violent'] / data['citypop'] * 100
# 显示更新后的DataFrame
# print(data)
# data.to_excel('6a.xlsx', index=False)



# 6b
# 假设你有一个名为'crime_data'的DataFrame，包含所有相关的变量
data.dropna(subset=['delta_ln_violent'], inplace=True)
data.dropna(subset=['sworn'], inplace=True)
# 创建包含自变量的设计矩阵
X = data[['sworn', 'violent','year']]

# 在模型中添加一个常数项
X = sm.add_constant(X)

# 因变量（堆叠的犯罪数据）
y = data['delta_ln_violent']

# 拟合线性回归模型
model = sm.OLS(y, X).fit()

# 打印回归结果
# print(model.summary())


# 7a

# 读取数据
data['delta_ln_sworn'] = np.log(data['sworn']) - np.log(data['sworn'].shift())
# 计算 ∆ ln (sta_welf)
data['∆ ln (sta_welf)'] = np.log(data['sta_welf']) - np.log(data['sta_welf'].shift())

# 计算 ∆ ln (sta_educ)
data['∆ ln (sta_educ)'] = np.log(data['sta_educ']) - np.log(data['sta_educ'].shift())

# 计算 ∆unemp
data['∆unemp'] = data['unemp'].diff()

# 计算 ∆citybla
data['∆citybla'] = data['citybla'].diff()

# 计算 ∆cityfemh
data['∆cityfemh'] = data['cityfemh'].diff()

# 计算 ∆a15_24
data['∆a15_24'] = (data['a15_19'] + data['a20_24']).diff()

# 使用get_dummies函数创建城市的虚拟变量
city_dummies = pd.get_dummies(data['city'], prefix='cc-')
# 将虚拟变量添加到原始DataFrame中
data = pd.concat([data, city_dummies], axis=1)

# 使用get_dummies函数创建城市的虚拟变量
# city_size_dummies = pd.get_dummies(data['citypop'], prefix='cs-')
# # 将虚拟变量添加到原始DataFrame中
# data = pd.concat([data, city_size_dummies], axis=1)

# 使用get_dummies函数创建城市的虚拟变量
year_dummies = pd.get_dummies(data['year'], prefix='year_')
# 将虚拟变量添加到原始DataFrame中
data = pd.concat([data, year_dummies], axis=1)


data = data.replace({True: 1, False: 0})

# 创建城市规模虚拟变量
data['cs1'] = (data['citypop'] < 250000).astype(int)
data['cs2'] = ((data['citypop'] >= 250000) & (data['citypop'] < 500000)).astype(int)
data['cs3'] = ((data['citypop'] >= 500000) & (data['citypop'] < 1000000)).astype(int)
data['cs4'] = (data['citypop'] >= 1000000).astype(int)
city_columns = data.columns[data.columns.str.startswith('cc-')].tolist()
# city_size_columns = data.columns[data.columns.str.startswith('cs-')].tolist()
year_columns = data.columns[data.columns.str.startswith('year_')].tolist()
data.dropna(subset=['∆unemp'], inplace=True)
data.dropna(subset=['∆cityfemh'], inplace=True)
data.dropna(subset=['delta_ln_sworn'], inplace=True)
# 设置自变量和因变量
X = data[['∆ ln (sta_educ)', '∆ ln (sta_welf)', '∆a15_24', '∆citybla', '∆cityfemh', '∆unemp','cs1', 'cs2', 'cs3', 'cs4'] + city_columns + year_columns]
y = data['delta_ln_sworn']

# 添加截距项
X = sm.add_constant(X)

# 拟合OLS回归模型
model = sm.OLS(y, X)
results = model.fit()

# 打印回归结果摘要
# print(results.summary())




# 7c

# 创建城市规模虚拟变量
# 设置自变量和因变量
X_first_stage = data[['elecyear', 'governor','cs1', 'cs2', 'cs3', 'cs4']+ city_columns + year_columns]
y_first_stage = data['delta_ln_sworn']

# 添加截距项
X_first_stage = sm.add_constant(X_first_stage)

# 拟合OLS回归模型
model_first_stage = sm.OLS(y_first_stage, X_first_stage)
results_first_stage = model_first_stage.fit()

# 保存拟合值（FITS）
data['d_sworn_FITS'] = results_first_stage.fittedvalues

# 输出回归结果摘要
# print(results_first_stage.summary())



# 8

# 创建第一个滞后值变量
data['d_sworn_FIT1'] = data['d_sworn_FITS'].shift(1)

# 创建第二个滞后值变量
data['d_sworn_FIT2'] = data['d_sworn_FITS'].shift(2)
data.dropna(subset=['d_sworn_FIT1'], inplace=True)
data.dropna(subset=['d_sworn_FIT2'], inplace=True)
# 设置自变量和因变量
X = data[['elecyear', 'governor', 'd_sworn_FIT1', 'd_sworn_FIT2','cs1', 'cs2', 'cs3', 'cs4']+ city_columns + year_columns]
y = data['delta_ln_sworn']

# 添加截距项
X = sm.add_constant(X)

# 拟合OLS回归模型
model = sm.OLS(y, X)
results = model.fit()

# 输出回归结果摘要
# print(results.summary())

# 8b


# 设置自变量和因变量
X = data[['elecyear', 'governor','cs1', 'cs2', 'cs3', 'cs4','d_sworn_FIT1', 'd_sworn_FIT2']+ city_columns + year_columns]
y = data['delta_ln_violent']  # 请确保这是您的因变量

# 添加截距项
X = sm.add_constant(X)

# 拟合OLS回归模型
model = sm.OLS(y, X)
results = model.fit()

# 输出回归结果摘要
print(results.summary())
