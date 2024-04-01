import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_stata('crime.dta')
# data['governor'] = data['governor'].fillna('NA')
# print(data)
# 1
data['delta_ln_sworn'] = np.log(data['sworn']) - np.log(data['sworn'].shift())
data['percent_change_sworn'] = (np.exp(data['delta_ln_sworn']) -1) * 100
# print(data)
# data.to_excel('percent_change.xlsx', index=False)


# 2
# 假设你有一个名为'data'的DataFrame，包含相关变量，包括:
# 'delta_ln_sworn': ∆ ln sworn（宣誓人员数量的对数差）
# 'mayoral_election_year': 表示市长选举年的虚拟变量（如果是则为1，否则为0）
# 'gubernatorial_election_year': 表示州长选举年的虚拟变量（如果是则为1，否则为0）
# 'year': 表示年份的虚拟变量

# 创建包含虚拟变量的设计矩阵

data.dropna(subset=['governor'], inplace=True)
data.dropna(subset=['delta_ln_sworn'], inplace=True)
data['elecyear'] = data['elecyear'].astype(int)
data['governor'] = data['governor'].astype(int)
data['year'] = data['year'].astype(int)



print(data)
X = data[['elecyear', 'governor', 'year']]

# 添加一个常数项到模型中
X = sm.add_constant(X)

# 因变量
y = data['delta_ln_sworn']

# 拟合线性回归模型
model = sm.OLS(y, X).fit()

# 打印回归结果摘要
# print(model.summary())

# 3a
# 假设你有一个名为'data'的DataFrame，包含每年的必要变量

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

# 显示更新后的包含新变量的DataFrame
# print(data)
# data.to_excel('ln_change.xlsx', index=False)

# 3b
# 假设你有一个名为'data'的DataFrame，包含城市人口变量'citypop'

# 创建城市规模虚拟变量
data['cs1'] = (data['citypop'] < 250000).astype(int)
data['cs2'] = ((data['citypop'] >= 250000) & (data['citypop'] < 500000)).astype(int)
data['cs3'] = ((data['citypop'] >= 500000) & (data['citypop'] < 1000000)).astype(int)
data['cs4'] = (data['citypop'] >= 1000000).astype(int)

# 显示更新后的DataFrame
# print(data)
# data.to_excel('3b.xlsx', index=False)

# 3c
# 假设你有一个名为'data'的DataFrame，包含所有必要的变量

# 创建包含所有自变量的设计矩阵
data.dropna(subset=['∆unemp'], inplace=True)
data.dropna(subset=['∆cityfemh'], inplace=True)
X = data[['elecyear', 'governor', 'year', 'cs1', 'cs2', 'cs3', 'cs4', 
          '∆ ln (sta_welf)', '∆ ln (sta_educ)', '∆unemp', '∆citybla', 
          '∆cityfemh', '∆a15_24']]

# 在模型中添加一个常数项
X = sm.add_constant(X)

# 因变量
y = data['delta_ln_sworn']

# 拟合线性回归模型
model = sm.OLS(y, X).fit()

# 打印回归结果
# print(model.summary())

# 4a
# 假设你有一个名为'data'的DataFrame，其中包含城市名称的列'city'

# 使用get_dummies函数创建城市的虚拟变量
city_dummies = pd.get_dummies(data['city'], prefix='city')

# 将虚拟变量添加到原始DataFrame中
data = pd.concat([data, city_dummies], axis=1)
data = data.replace({True: 1, False: 0})
# 显示更新后的DataFrame
# print(data)
# data.to_excel('4a.xlsx', index=False)

# 4b
# 假设你有一个名为'data'的DataFrame，包含所有必要的变量
city_columns = data.columns[data.columns.str.startswith('city_')].tolist()
# 创建包含所有自变量的设计矩阵
X = data[['elecyear', 'governor', 'year', 'cs1', 'cs2', 'cs3', 'cs4', 
          '∆ ln (sta_welf)', '∆ ln (sta_educ)', '∆unemp', '∆citybla', 
          '∆cityfemh', '∆a15_24'] + city_columns]

# 在模型中添加一个常数项
X = sm.add_constant(X)

# 因变量
y = data['delta_ln_sworn']

# 拟合线性回归模型
model = sm.OLS(y, X).fit()

# 打印回归结果
print(model.summary())
