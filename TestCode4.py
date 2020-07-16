#  axis=1,跨列，axis=0，跨行

# ----------------------------------------------------import相关包和用户函数--------------------------------------------#
import numpy as np
import pandas as pd

# pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
# pd.set_option('display.width', 1000)

# 导入可视化画图的包matplotlib
import matplotlib.pyplot as plt
# ggplot是matplotlib中的一种好看的画图包
plt.style.use('ggplot')

# 导入可视化画图的包Seaborn
import seaborn as sns

# 为了减少各种warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 自行实现的一个画出不同分类模型的learning curves
from plot_learning_curve import plot_learning_curve

# 导入检测离群点函数detect_outliers
from detect_outliers import detect_outliers

# 导入填补缺失Age得函数
from set_missing_ages import set_missing_ages


# --------------------------------------------读入数据------------------------------------------------------------------#
# 加载训练集test和测试集test为dataFrame格式
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# 显示train的列名和具体数值信息等等
print(train.head())

# 显示每一列的详细信息，包括数值类型、数值总数、文件大小等
train.info()
train.describe()


# --------------------------------------------------检测离群值--------------------------------------------------------#
# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
train.loc[Outliers_to_drop]

# 通过drop函数，把离群值（行做索引为Outliers_to_drop）去除
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

train.info()
train.describe()

# -------------------------------缺失值分析------------------------------------------#
# 利用fillna函数填补缺失值为NaN
train = train.fillna(np.nan)
test = test.fillna(np.nan)

# 观察train set 和 Test set各自的具体统计数据
print("============= Train Set Info ================")
print(train.info())
print("============= Test Set Info =================")
print(test.info())

# 观察train set 和 Test set各自的缺失值统计数据
print("=========== Missing Values Train =============")
print(train.isnull().sum())
print("=========== Missing Values Test ==============")
print(test.isnull().sum())



# -----------------------------------------------填补缺失值-------------------------------------------------------------#
# isnull和notnull都会根据是否为空把矩阵变为true/false
# 显示'Cabin'那一列的每个具体数值
print(train["Cabin"][train["Cabin"].notnull()].head())

# 处理train和test data中的'Cabin'缺失值
def fix_cabin(data):
    # Series函数(Series为DataFrame中的一列这个意思)，把一个单列list转化为带列索引index的双列list
    data["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in data['Cabin'] ])
fix_cabin(train)
fix_cabin(test)
print("=============After impute 'Cabin'==============")
print(train.head())


# 处理train和test data中的除'Cabin'之外的缺失值
def fill_missing_values(df):
    ''' 对于数值型缺失值。用中位数填充；对于非数值型缺失值，用众数填充'''
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for column in list(missing.index):
        if df[column].dtype == 'object': # 非数值类型
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':# 数值类型
            df[column].fillna(df[column].median(), inplace=True)

fill_missing_values(train)
fill_missing_values(test)

# # 利用自定义的函数填补缺失值
# train, rfr = set_missing_ages(train)
# # 用同样的RandomForestRegressor模型填上丢失的年龄
# tmp_df = test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
# null_age = tmp_df[test.Age.isnull()].values
# # 根据特征属性X预测年龄并补上
# X = null_age[:, 1:]
# predictedAges = rfr.predict(X)
# test.loc[ (test.Age.isnull()), 'Age' ] = predictedAges


# 观察train set 和 Test set各自的缺失值统计数据
print("=== Missing Values Train After Filling Missings ===")
print(train.isnull().sum())
print("=== Missing Values Test After Filling Missings ===")
print(test.isnull().sum())

print("=== Train info After Filling Missings ===")
print(train.info())



#-----------------------------------------------------标准化--------------------------------------------------------- #
# # 对Fare列归一化
# max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
# train[['Fare']].apply(max_min_scaler)




# ------------------------探索不同特征属性(SibSp、 Parch、 Age 、Fare values等)与Survived之间的关系----------------------#
# 数值型属性(SibSp、 Parch、 Age 、Fare values) 和 Survived之间的相关矩阵。（没有Cabin\Embarked\Sex）
# loc[]列，ilo[]行
g = sns.heatmap(train.iloc[:, 1:].corr(),
                        annot=True, # true为每个方格写入数据
                        fmt=".3f", # 格式设置
                        cmap="coolwarm") # cool/ coolwarm
# plt.show()

# SibSp feature vs Survived
g = sns.catplot(x="SibSp", y="Survived", data=train, kind="bar", height=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
# plt.show()


# Parch feature vs Survived
g = sns.catplot(x="Parch", y="Survived", data=train, kind="bar", height=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
# plt.show()

# Age feature vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
# plt.show()

# Age distibution
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade=True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax=g, color="Blue",
                        shade=True)
# plt.show()

g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived", "Survived"])
# plt.show()

# Sex feature vs Survived
g = sns.barplot(x="Sex", y="Survived", data=train)
g = g.set_ylabel("Survival Probability")
# plt.show()

# Pclass vs Survived
g = sns.catplot(x="Pclass",y="Survived",data=train,kind="bar", height = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
# plt.show()

# Pclass vs Embarked on Survived
g = sns.catplot("Pclass", col="Embarked",  data=train,
                   height=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")
# plt.show()

# 不同字母开头的Cabin的存活率（X为缺失的Cabin数据）
g = sns.countplot(train["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
# plt.show()
g = sns.catplot(y="Survived",x="Cabin",data=train,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")
# plt.show()

# ---------------------------------------------------特征工程-------------- -----------------------------------------#

# 从Name特征属性提取称呼(Title)
def get_title(data):
    data_title = [i.split(",")[1].split(".")[0].strip() for i in data["Name"]] # 提取并且构建称呼Title
    data["Title"] = pd.Series(data_title) # 把'Title'作为一个列添加到DataFrame对象data中

get_title(train)
get_title(test)
print (train['Title'].head())
print (test['Title'].head())

# 展示不同Title的人的数量
g = sns.countplot(x="Title",data=train)
g = plt.setp(g.get_xticklabels(), rotation=45)
# plt.show()

# # ------------------------ Name & Title------------------------- #
# # 把'Title'中的转化为数值
# def convert_title (data) :
#     # 把'Title'中的稀少的Title设置为Rare
#     data["Title"] = data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col',
#                                              'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#     data["Title"] = data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
#     data["Title"] = data["Title"].astype(int) # 修改Title的数值类型为0、1、2、3四种
#
# convert_title(train)
# convert_title(test)
#
# g = sns.catplot(x="Title",y="Survived",data=train,kind="bar")
# g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
# g = g.set_ylabels("survival probability")
# # plt.show()


# ------------------- Name & Ttile -------------------- #

train['Title']=train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
test['Title']=test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())


newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}


train['Title']=train.Title.map(newtitles)
test['Title']=test.Title.map(newtitles)

# 画图
g = sns.catplot(x="Title",y="Survived",data=train,kind="bar")
g = g.set_xticklabels(["Officer","Royalty","Mrs","Miss", "Mr", "Master"])
g = g.set_ylabels("survival probability")
plt.show()

print ('----------------after map new titles ----------')
print (train.head())

# # one hot 编码 (下面才统一进行)
# train = pd.get_dummies(train, columns = ["Title"])
# test = pd.get_dummies(test, columns = ["Title"])


# # ------------------- Name & Ttile & Sex & Age -------------------- #
# def newage (cols):
#     title=cols[0]
#     Sex=cols[1]
#     Age=cols[2]
#     if pd.isnull(Age):
#         if title=='Master' and Sex=="male":
#             return 4.57
#         elif title=='Miss' and Sex=='female':
#             return 21.8
#         elif title=='Mr' and Sex=='male':
#             return 32.37
#         elif title=='Mrs' and Sex=='female':
#             return 35.72
#         elif title=='Officer' and Sex=='female':
#             return 49
#         elif title=='Officer' and Sex=='male':
#             return 46.56
#         elif title=='Royalty' and Sex=='female':
#             return 40.50
#         else:
#             return 42.33
#     else:
#         return Age
#
# train.Age=train[['title','Sex','Age']].apply(newage, axis=1)
# test.Age=test[['title','Sex','Age']].apply(newage, axis=1)




# ------------------------Name------------------------- #
# Drop Name Column
train.drop(labels = ["Name"], axis = 1, inplace = True)
test.drop(labels=["Name"], axis=1, inplace=True)



# --------------------Ticket------------------------ #
train['Ticket2'] = train.Ticket.apply(lambda x : len(x))
test['Ticket2'] = test.Ticket.apply(lambda x : len(x))
sns.barplot('Ticket2','Survived',data=train)
plt.show()



# --------------------Family Size---------------------- #
# 新的特征向量family size，家庭成员总数（包括本人）
train["Fsize"] = train["SibSp"] + train["Parch"] + 1
test["Fsize"] = test["SibSp"] + train["Parch"] + 1

g = sns.catplot(x="Fsize",y="Survived",data=train, kind='bar')
g = g.set_ylabels("Survival Probability")
# plt.show()

# Create new feature of family size
def create_fsize(data):
    data['Single'] = data['Fsize'].map(lambda s: 1 if s == 1 else 0)
    data['SmallF'] = data['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    data['MedF'] = data['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    data['LargeF'] = data['Fsize'].map(lambda s: 1 if s >= 5 else 0)

create_fsize(train)
create_fsize(test)

g = sns.catplot(x="Single",y="Survived",data=train,kind="bar")
g = g.set_ylabels("Survival Probability")
# plt.show()
g = sns.catplot(x="SmallF",y="Survived",data=train,kind="bar")
g = g.set_ylabels("Survival Probability")
# plt.show()
g = sns.catplot(x="MedF",y="Survived",data=train,kind="bar")
g = g.set_ylabels("Survival Probability")
# plt.show()
g = sns.catplot(x="LargeF",y="Survived",data=train,kind="bar")
g = g.set_ylabels("Survival Probability")
# plt.show()

print ('------------train head after Fsize settings---------')
print(train.head())


# --------------------Embarked & Title---------------------- #
def encode_train_embark(data):
    '''把特征属性Embarked转化为数值向量并且加入到DataFrame data中 '''
    data = pd.get_dummies(data, columns = ["Title"]) # get_dummies 实现one hot encode（即 0 0 1; 0 1 0; 1 0 0;...）
    data = pd.get_dummies(data, columns = ["Embarked"], prefix="Em")

encode_train_embark(train)
encode_train_embark(test)

# --------------------Cabin------------------------- #
train['Cabin'] = train['Cabin'].map({'X': 1, 'C': 2, 'B': 3, 'D': 4, 'E': 5, 'A': 6, 'F': 7, 'G': 8, 'T': 9})
test['Cabin'] = test['Cabin'].map({'X': 1, 'C': 2, 'B': 3, 'D': 4, 'E': 5, 'A': 6, 'F': 7, 'G': 8, 'T': 9})

# --------------------Pclass------------------------- #
train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")
test["Pclass"] = test["Pclass"].astype("category")
test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")


# -------------"PassengerId", "Ticket"--------------- #
train.drop(labels = ["PassengerId", "Ticket"], axis = 1, inplace=True)
test.drop(labels = ["PassengerId", "Ticket"], axis=1, inplace=True)


print ('------------train head after Embarked / Cabin / Pclass---------')
print(train.head())

from sklearn.preprocessing import LabelEncoder
def impute_cats(df):
    '''ML算法只接受数值型输入，把非数值型或类别性质的特征属性如Sex/Embarked等转化为数值型'''
    # Find the columns of object type along with their column index
    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    object_cols_ind = []
    for col in object_cols:
        object_cols_ind.append(df.columns.get_loc(col))

    # Encode the categorical columns with numbers
    label_enc = LabelEncoder()
    for i in object_cols_ind:
        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])

# 把非数值型或类别性质的特征属性如Sex/Embarked等转化为数值型
impute_cats(train)
impute_cats(test)
print("Train Dtype counts: \n{}".format(train.dtypes.value_counts()))
print("Test Dtype counts: \n{}".format(test.dtypes.value_counts()))


# # 看了那个姐们的博客，我尝试drop一些，看看准确率是否提高了;
# # 别drop，那姐们是已经提取了SibSp / Parch / Cabin 的信息才drop
# train.drop(['SibSp','Parch','Cabin'],axis=1,inplace=True)
# test.drop(['SibSp','Parch','Cabin'],axis=1,inplace=True)

train.info()
test.info()

print ('------------train head after feature engineering---------')
print(train.head())

g = sns.heatmap(train.iloc[:, 0:].corr(),
                        annot=False, # true为每个方格写入数据
                        fmt="f", # 格式设置
                        cmap="coolwarm") # cool/ coolwarm
plt.show()



# ---------------------------------------------------模型选择-------------- -----------------------------------------#
# import the models
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# import evaluators, stacking, etc.
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
# Package for stacking models
from vecstack import stacking
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# train = 特征向量X + 标签y/回归值y
y = train['Survived']
X = train.drop('Survived', axis=1)


def train_model(classifier, name="Classifier"):
    '''这个函数是用户训练所有模型并且打印每个模型的accuracy'''

    # 分层采样，KFold不同的是，保证不同类别的样本在训练集和测试集的比例一致
    folds = StratifiedKFold(n_splits=5, random_state=42)
    accuracy = np.mean(cross_val_score(classifier, X, y, scoring="accuracy", cv=folds, n_jobs=-1)) # 求取所有“折”的accuracy均值
    if name not in alg_list:
        alg_list.append(name) # 把算法名字加到List对象alg_list中
    print(f"{name} Accuracy: {accuracy}")
    return accuracy

# 交叉值平均值和算法名称的列表
cv_means = []
alg_list = []


''' 以下是多种模型的比较，共8个。注意.fit(X, y)返回模型变量然后可以用模型变量来做预测.predict(X_test)'''
# --------------------LogisticRegression------------------------ #
# Initialize the model
#         LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
log_reg = LogisticRegression(C=5, penalty='l1',tol=1e-6, random_state=42) # penalty 有l1和l2
# Validate the model
log_reg_acc = train_model(log_reg, "Logistic Regression")
cv_means.append(log_reg_acc)
# Fit the best performing model to training data
log_reg.fit(X, y)



# --------Support Vector Machine without GridSeachCV------------ #
# Initialize the model
svm = SVC(C=5, random_state=42)
# Validate the model
svm_acc = train_model(svm, "Support Vector Machine")
cv_means.append(svm_acc)
# Fit the best performing model to training data
svm.fit(X, y)



# --------------------RandomForestClassifier--------------------- #
# Initialize the model
rf = RandomForestClassifier(n_estimators=50, max_depth=20,
                                min_samples_split=2, min_samples_leaf=5,
                                max_features="log2", random_state=12)
# Validate the model
rf_acc = train_model(rf, "Random Forest")
cv_means.append(rf_acc)
# Fit the best performing model to training data
rf.fit(X, y)



# -----------------LinearDiscriminantAnalysis-------------------- #
# Initialize the model
lda = LinearDiscriminantAnalysis(solver='lsqr')
# Validate the model
lda_acc = train_model(lda, "Linear Discriminant Analysis")
cv_means.append(lda_acc)
# Fit the best performing model to training data
lda.fit(X, y)


# --------------------------MLPClassifier------------------------ #
# Initialize the model
mlp = MLPClassifier(hidden_layer_sizes=(50, 10), activation='relu', solver='adam',
                    alpha=0.01, batch_size=32, learning_rate='constant',
                    shuffle=False, random_state=42, early_stopping=True,
                    validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
# Validate the model
mlp_acc = train_model(mlp, "MLP")
cv_means.append(mlp_acc)
# Fit the best performing model to training data
mlp.fit(X, y)


# --------------------------XGBClassifier------------------------ #
# Initialize the model
xgb = XGBClassifier(max_depth=5, learning_rate=0.1, n_jobs=-1, nthread=-1,
                    gamma=0.06, min_child_weight=5,
                    subsample=1, colsample_bytree=0.9,
                    reg_alpha=0, reg_lambda=0.5,
                    random_state=42)
# Validate the model
xgb_acc = train_model(xgb, "XgBoost")
cv_means.append(xgb_acc)
# Fit the best performing model to training data
xgb.fit(X, y)



# --------------------------LGBMClassifier----------------------- #
# Initialize the model
lgbm = LGBMClassifier(num_leaves=31, learning_rate=0.1,
                      n_estimators=64, random_state=42, n_jobs=-1)
# Validate the model
lgbm_acc = train_model(lgbm, "LGBM")
cv_means.append(lgbm_acc)
# Fit the best performing model to training data
lgbm.fit(X, y)



# ---------------------GSRF (GridSeachCV RandomForest)----------- #
# Initialize the model
RF = RandomForestClassifier(random_state=1)
PRF = [{'n_estimators':[50,350],'max_depth':[15,30],'criterion':['gini','entropy'], 'min_samples_leaf':[2, 10]}]
gsrf = GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy', cv=2)
# Validate the model
gsrf_acc = train_model(gsrf, "GridSeachCV-RandomForest")
cv_means.append(gsrf_acc)
# Fit the best performing model to training data
gsrf.fit(X, y)



# ------------------GSSVM (GridSeachCV SVM)----------------------- #
gssvc = make_pipeline(StandardScaler(), SVC(random_state=1))
r = [0.0001,0.001,0.1,1,10,50,100]
PSVM = [{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
gssvm = GridSearchCV(estimator = gssvc, param_grid=PSVM, scoring='accuracy', cv=2)
# Validate the model
gssvm_acc = train_model(gssvm, "GridSeachCV-SVM")
cv_means.append(gssvm_acc)
# Fit the best performing model to training data
gssvm.fit(X, y)
''' 以上是多种模型的比较，注意.fit(X, y)返回模型变量然后可以用模型变量来做预测.predict(X_test)'''



# 展示所有模型交叉验证的平均accuracy
performance_df = pd.DataFrame({"Algorithms": alg_list, "CrossValMeans":cv_means})
g = sns.barplot("CrossValMeans","Algorithms", data = performance_df, palette="Set3",orient = "h")
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
plt.show()



# --------------------------------------------------Stacking-------------------------------------------------------- #
''' 以下是多种模型的stacking'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 第一层模型
models = [rf, lda, lgbm, log_reg, mlp, gssvm, gsrf]
# Perform Stacking
print ('-----------------performing stacking-------------------')
S_train, S_test = stacking(models,
                           X_train, y_train, X_test,
                           regression=False,
                           mode='oof_pred_bag',
                           n_folds = 5,
                           save_dir=None,
                           needs_proba=False,
                           random_state=42,
                           stratified=True,
                           shuffle=True,
                           verbose=2
                          )

# 进入第二层，利用第一层的输出值S_train
xgb.fit(S_train, y_train)

# 在S_test上做预测，作为最终的accuracy
stacked_pred = xgb.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, stacked_pred))
''' 以上是多种模型的stacking'''



# -----------------------------------------最终的在未知分类的的test上进行预测------------------------------------------- #
# 注意：要和models[]中的基本模型一致, models中最多7种
# models = [rf, lda, lgbm, log_reg, mlp, gssvm, gsrf]
y1_pred_L1 = models[0].predict(test)
y2_pred_L1 = models[1].predict(test)
y3_pred_L1 = models[2].predict(test)
y4_pred_L1 = models[3].predict(test)
y5_pred_L1 = models[4].predict(test)
y6_pred_L1 = models[5].predict(test)
y7_pred_L1 = models[6].predict(test)
S_test_L1 = np.c_[y1_pred_L1, y2_pred_L1, y3_pred_L1, y4_pred_L1, y5_pred_L1, y6_pred_L1, y7_pred_L1] # np.c_:按行转化为矩阵


# models = [rf, lda, lgbm, log_reg, mlp, gssvm, gsrf]
# 求取stacking在test上的accuracy
test_stacked_pred = xgb.predict(S_test_L1) # xgb是stacking后的xgb

# 求取原来xgbclassifier在test上的accuracy
xgb.fit(X, y)
xgb_pred = xgb.predict(test)

# 求取原来RandomForest在test上的accuracy
rf_pred = rf.predict(test)

# 求取原来LinearDiscriminantAnalysis在test上的accuracy
lda_pred = lda.predict(test)

# 求取原来LGBMClassifier在test上的accuracy
lgbm_pred = lgbm.predict(test)

# 求取原来LogisticRegression在test上的accuracy
log_reg_pred = log_reg.predict(test)

# 求取原来MLPClassifier在test上的accuracy
mlp_pred = mlp.predict(test)

# 求取原来GridSearch SVM在test上的accuracy
gssvm_pred = gssvm.predict(test)

# 求取原来GridSearch RandomForest在test上的accuracy
gsrf_pred = gsrf.predict(test)




# --------------------------------------------------预测结果写入文件-------------------------------------------------- #
# models = [rf, lda, lgbm, log_reg, mlp, gssvm, gsrf]
old_test = pd.read_csv('../input/test.csv')

# 构建预测结果为DataFrame格式
submission = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': test_stacked_pred})
xgb_sub = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': xgb_pred})
rf_sub = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': rf_pred})
lda_sub = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': lda_pred})
lgbm_sub = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': lgbm_pred})
log_reg_sub = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': log_reg_pred})
mlp_sub = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': mlp_pred})
gssvm_sub = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': gssvm_pred})
gsrf_sub = pd.DataFrame({'PassengerId':old_test['PassengerId'], 'Survived': gsrf_pred})


submission.to_csv("../Titanic4/output/stacked_submission.csv", index=False)
xgb_sub.to_csv("../Titanic4/output/xgboost_submission.csv", index=False)
rf_sub.to_csv("../Titanic4/output/random_forest_submission.csv", index=False)
lda_sub.to_csv("../Titanic4/output/lda_submission.csv", index=False)
lgbm_sub.to_csv("../Titanic4/output/lgbm_submission.csv", index=False)
log_reg_sub.to_csv("../Titanic4/output/log_reg_submission.csv", index=False)
mlp_sub.to_csv("../Titanic4/output/mpl_submission.csv", index=False)
gssvm_sub.to_csv("../Titanic4/output/gssvm_submission.csv", index=False)
gsrf_sub.to_csv("../Titanic4/output/gsrf_submission.csv", index=False)



# best_score = pd.read_csv('../input/best_score (1).csv')
# best_score.to_csv('best_score.csv', index=False)


# ------------------------------------------------画出每个模型的学习曲线-------------------------------------------------- #
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)


# models = [rf, lda, lgbm, log_reg, mlp, gssvm, gsrf]
plot_learning_curve(xgb, u"XGB Leaning Curve", X, y)
plot_learning_curve(rf, u"Random Forest Leaning Curve", X, y)
plot_learning_curve(lda, u"LDA Leaning Curve", X, y)
plot_learning_curve(lgbm, u"LGBM Leaning Curve", X, y)
plot_learning_curve(log_reg, u"LogisticRegression Leaning Curve", X, y)
plot_learning_curve(mlp, u"MLP Leaning Curve", X, y)
plot_learning_curve(gssvm, u"GridSearch SVM Leaning Curve", X, y)
plot_learning_curve(gsrf, u"GridSearch Random Forest Leaning Curve", X, y)













