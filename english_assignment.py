# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:00:28 2025

@author: Chen_Lab01
"""

import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# read file 
df = pd.read_excel('DATA_Kiss_count_gender_and_IQ.xlsx')  # <- 替換成你自己的檔名

#  (Male=1, Female=0)
df['Gender_Num'] = df['Gender'].map({'male': 1, 'female': 0})

# 基礎描述統計
print(df.describe())

# 計算皮爾森相關係數 (r)、t值與p值
def correlation_stats(x, y):
    r, p = stats.pearsonr(x, y)
    df_n = len(x) - 2
    t = r * ((df_n / (1 - r**2)) ** 0.5)
    return r, t, p

# 對照組合：test true or not
pairs = [
    ('Gender_Num', 'Kiss Count'),
    ('Gender_Num', 'Age of First Kiss'),
    ('Gender_Num', 'IQ'),
    ('IQ', 'Kiss Count'),
    ('IQ', 'Age of First Kiss'),
]

for x, y in pairs:
    r, t, p = correlation_stats(df[x], df[y])
    print(f'Correlation between {x} and {y}: r={r:.4f}, t={t:.4f}, p={p:.4f}')

# Step 5: 繪圖 - 散佈圖與趨勢線
sns.pairplot(df[['Gender_Num', 'IQ', 'Kiss Count', 'Age of First Kiss']])
plt.suptitle("Pairwise Plots", y=1.02)
plt.show(block = True)

# Step 6: 額外 - 性別分組統計
group_stats = df.groupby('Gender')[['IQ', 'Kiss Count', 'Age of First Kiss']].mean()
print("\nGrouped Statistics by Gender:")
print(group_stats)


# 正確數據輸入
gender_means = pd.DataFrame({
    'Male': [11.0, 113.5, 18.71],
    'Female': [9.1, 109.3, 19.87]
}, index=['Kiss Count', 'IQ', 'Age of First Kiss'])

# 繪圖

df = pd.read_excel("DATA_Kiss_count_gender_and_IQ.xlsx")

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Gender', y='Kiss Count', palette='Set2')
plt.xticks([0, 1], ['Male', 'Female'])
plt.xlabel('Gender')
plt.ylabel('Kiss Count')
plt.title('Distribution of Kiss Count by Gender')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.regplot(data=df, x='Kiss Count', y='Age of First Kiss', scatter_kws={'alpha':0.5})
plt.xlabel('Kiss Count')
plt.ylabel('Age of First Kiss')
plt.title('Kiss Count vs Age of First Kiss')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.lmplot(data=df, x='Kiss Count', y='IQ', hue='Gender', palette='Set1', scatter_kws={'alpha':0.5})
plt.xlabel('Kiss Count')
plt.ylabel('IQ')
plt.title('Kiss Count vs IQ (by Gender)')
plt.tight_layout()
plt.show()

gender_means.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'pink'])
plt.title('Mean Comparison by Gender')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# 把 Gender 轉為數值
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

# 計算相關矩陣
corr = df[['Gender', 'Kiss Count', 'IQ', 'Age of First Kiss']].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
