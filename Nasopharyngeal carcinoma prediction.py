#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV


train_data = pd.read_excel(r'E:\shuju\rain_group.xlsx')
val_data = pd.read_excel(r'E:\shuju\valid_group.xlsx')


X_train = train_data.drop(columns=['label'])  
y_train = train_data['label']

X_val = val_data.drop(columns=['label']) 
y_val = val_data['label']

# 1. T test
t_stat, p_val = stats.ttest_ind(train_data['label'], val_data['label'])  

print(f'T-test Statistic: {t_stat}, p-value: {p_val}')

# 2. LASSO regression and 10-fold cross-validation
lasso = LassoCV(cv=10)  
lasso.fit(X_train, y_train)

print(f'Best LASSO Regularization Parameter Alpha: {lasso.alpha_}')


alpha_list = lasso.alphas_  
coef = lasso.coef_path_  


plt.figure(figsize=(12, 8))


for i in range(coef.shape[0]):
    plt.plot(np.log(alpha_list), coef[i], label=f'Feature {i+1}')  

plt.title('LASSO Coefficients Convergence Plot')
plt.xlabel('log(λ)')
plt.ylabel('Coefficient values')
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.legend()
plt.grid()
plt.show()


val_predictions = lasso.predict(X_val)

coef = lasso.coef_
feature_names = X_train.columns


best_feature_indices = np.argsort(np.abs(coef))[-7:]  
best_features = feature_names[best_feature_indices]
best_coef = coef[best_feature_indices]

print(f'Best Radiomics Features: {best_features}')
print(f'Corresponding Coefficients: {best_coef}')

cv_scores = cross_val_score(lasso, X_train, y_train, cv=10)

print(f'Cross-validation Score (Accuracy): {np.mean(cv_scores)}')


plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores, marker='o', linestyle='-', color='b')
plt.title('Cross-Validation Scores for Best Features')
plt.xlabel('Fold')
plt.ylabel('Cross-Validation Score')
plt.ylim(0, 1)  
plt.axhline(np.mean(cv_scores), color='r', linestyle='--', label='Mean Score')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x=best_features, y=best_coef)
plt.title('LASSO Regression Coefficients for Best Features')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.xticks(rotation=45)
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
for feature in best_features:
    sns.boxplot(x='target', y=feature, data=train_data)
    plt.title(f'Boxplot of {feature} by Target')
    plt.xlabel('Target')
    plt.ylabel(feature)
    plt.grid()
    plt.show()


# In[ ]:


# Rad-score
def compute_rad_score(data):
    return (0.02416 * data['originalngtdmCoarseness'] +
            0.19997 * data['wavelet-LHLfirstorderMaximum'] +
            0.03121 * data['wavelet-LHHglcmClusterShade'] +
            0.19396 * data['wavelet-HLHglcmIdmn'] +
            0.18199 * data['wavelet-HLHgldmSmallDependenceLowGrayLevelEmphasis'] +
            0.14896 * data['wavelet-HHLfirstorderMinimum'] +
            0.21974 * data['wavelet-HHLglcmJointAverage'])


train_data['Rad_score'] = compute_rad_score(train_data)
valid_data['Rad_score'] = compute_rad_score(valid_data)


means = [train_data['Rad_score'].mean(), valid_data['Rad_score'].mean()]
stds = [train_data['Rad_score'].std(), valid_data['Rad_score'].std()]
groups = ['Training Group', 'Validation Group']


plt.figure(figsize=(12, 6))
bar_width = 0.4
index = np.arange(len(groups))


bar1 = plt.bar(index, means, bar_width, yerr=stds, capsize=5, color='b', alpha=0.7, label='Mean Rad-score')


plt.xlabel('Group')
plt.ylabel('Rad-score')
plt.title('Mean Rad-score with Standard Deviation by Group')
plt.xticks(index, groups)
plt.ylim(0, max(means) + max(stds) + 0.1)  
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.legend()
plt.grid(axis='y')


plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=[train_data['Rad_score'], valid_data['Rad_score']], palette="Set3")
plt.xticks([0, 1], ['Training Group', 'Validation Group'])
plt.title('Boxplot of Rad-score by Group')
plt.ylabel('Rad-score')
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
sns.violinplot(data=[train_data['Rad_score'], valid_data['Rad_score']], palette="Set3")
plt.xticks([0, 1], ['Training Group', 'Validation Group'])
plt.title('Violin Plot of Rad-score by Group')
plt.ylabel('Rad-score')
plt.grid()
plt.show()


# In[ ]:


import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve


features = [
    'originalngtdmCoarseness',
    'wavelet-LHLfirstorderMaximum',
    'wavelet-LHHglcmClusterShade',
    'wavelet-HLHglcmIdmn',
    'wavelet-HLHgldmSmallDependenceLowGrayLevelEmphasis',
    'wavelet-HHLfirstorderMinimum',
    'wavelet-HHLglcmJointAverage'
]

X_train = train_data[features]
y_train = train_data['label']  
X_valid = valid_data[features]
y_valid = valid_data['label']

# build XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'gamma': [0, 0.1, 0.2],
}


grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)


print("Best parameters found: ", grid_search.best_params_)


best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)


y_pred_prob = best_model.predict_proba(X_valid)[:, 1]
y_pred = best_model.predict(X_valid)


auc = roc_auc_score(y_valid, y_pred_prob)
accuracy = accuracy_score(y_valid, y_pred)

tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
sensitivity = tp / (tp + fn)  
specificity = tn / (tn + fp) 

print(f"AUC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# the ROC curve
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[ ]:


# the DCA curve

def dca(y_true, y_prob, threshold):
    
    positive_count = np.sum(y_true)
    
    negative_count = len(y_true) - positive_count

   
    net_benefit = (y_prob >= threshold) * positive_count - (y_prob >= threshold) * negative_count * (threshold / (1 - threshold))
    return np.mean(net_benefit)



y_pred_prob_train = best_model.predict_proba(train_data[features])[:, 1]
y_pred_prob_valid = best_model.predict_proba(valid_data[features])[:, 1]
y_train = train_data['label']  
y_valid = valid_data['label']

thresholds = np.linspace(0, 1, 100)
train_net_benefits = []
valid_net_benefits = []

for threshold in thresholds:
    train_net_benefit = dca(y_train.values, y_pred_prob_train, threshold)
    valid_net_benefit = dca(y_valid.values, y_pred_prob_valid, threshold)
    train_net_benefits.append(train_net_benefit)
    valid_net_benefits.append(valid_net_benefit)


plt.figure(figsize=(10, 6))
plt.plot(thresholds, train_net_benefits, label='Training Group', color='blue')
plt.plot(thresholds, valid_net_benefits, label='Validation Group', color='orange')

plt.plot(thresholds, thresholds * np.sum(y_train) / len(y_train), label='All Positive', color='green', linestyle='--')
plt.plot(thresholds, np.zeros_like(thresholds), label='All Negative', color='red', linestyle='--')

plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis')
plt.legend()
plt.grid()
plt.xlim(0, 1)
plt.ylim(-0.1, 0.1)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.show()


# In[ ]:


#Delong test

from sklearn.model_selection import train_test_split
from scipy import stats
import statsmodels.api as sm
from mpl_toolkits.axes_grid1 import make_axes_locatable


y_valid = valid_data['label'] 
y_pred_prob_model1 = best_model.predict_proba(valid_data[features])[:, 1]
y_pred_prob_model2 = model2.predict_proba(valid_data[features])[:, 1]

auc_model1 = roc_auc_score(y_valid, y_pred_prob_model1)
auc_model2 = roc_auc_score(y_valid, y_pred_prob_model2)

fpr1, tpr1, _ = roc_curve(y_valid, y_pred_prob_model1)
fpr2, tpr2, _ = roc_curve(y_valid, y_pred_prob_model2)

def delong_roc_test(y_true, y_pred1, y_pred2):
    """ Perform DeLong Test for AUC comparison """
    from sklearn.metrics import roc_auc_score
    from statsmodels.tools.tools import add_constant
    
    n1 = len(y_pred1)
    n2 = len(y_pred2)
    
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    
    cov12 = np.cov(y_pred1, y_pred2)
    
    z_statistic = (auc1 - auc2) / np.sqrt(cov12[0][0]/n1 + cov12[1][1]/n2 - 2*cov12[0][1]/np.sqrt(n1*n2))
    return z_statistic

z_statistic = delong_roc_test(y_valid, y_pred_prob_model1, y_pred_prob_model2)
p_value = stats.norm.sf(abs(z_statistic)) * 2  

plt.figure(figsize=(10, 8))
plt.plot(fpr1, tpr1, color='blue', label=f'Model 1 (AUC = {auc_model1:.2f})')
plt.plot(fpr2, tpr2, color='orange', label=f'Model 2 (AUC = {auc_model2:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  

plt.text(0.5, z_statistic, f'Z-statistic: {z_statistic:.2f}', fontsize=12, color='black')

plt.xticks([0, 1], ['Training Set', 'Validation Set'])
plt.ylabel('Test Z-statistic')
plt.title('ROC Curve and DeLong Test Z-statistic')
plt.legend()
plt.grid()
plt.xlim([0.0,1.0])
plt.ylim([0, np.max([z_statistic + 1, 3])])  
plt.show()

