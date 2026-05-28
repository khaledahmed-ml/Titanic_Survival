import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
#------------------------------------------------------------------
df = pd.read_csv(r"F:\study programming\Python\Machine Learning\Task4\Titanic_train.csv")

df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Embarked', 'Ticket'])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].astype("category").cat.codes
df = df.dropna()

X = df.drop(columns='Survived')
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=10000,penalty='l2',C=.3,solver='lbfgs').fit(X_train_scaled, Y_train)
#-----------------------------
# Score
print(f"The Accuracy of Train: {log_reg.score(X_train_scaled, Y_train)}")
print(f"The Accuracy of Test: {log_reg.score(X_test_scaled,  Y_test)}")
print('#'*50)
#-----------------------------
# Coef & intercept
print(f" w= {log_reg.coef_}")
print(f" b= {log_reg.intercept_}")
print('#'*50)
#-----------------------------
will_you_survive = pd.DataFrame([{
    'Pclass': 2,
    'Sex':    0,
    'Age':    25,
    'SibSp':  0,
    'Parch':  0,
    'Fare':  70 
}])

will_you_survive_scaled = scaler.transform(will_you_survive)
prediction  = log_reg.predict(will_you_survive_scaled)
probability = log_reg.predict_proba(will_you_survive_scaled)

print(f"The Prediction         : {'Survived' if prediction[0] == 1 else 'Not Survived'}")
print(f"Probability of survival: {probability[0][1]:.2%}")
print(f"Possibility of death   : {probability[0][0]:.2%}")
print('#'*50)
#-----------------------------
#classification_report
Y_pred = log_reg.predict(X_test_scaled)
print(classification_report(Y_test, Y_pred, target_names=['Died', 'Survived']))
#----------------------------
#Confusion Matrix
cm=confusion_matrix(Y_test,log_reg.predict(X_test_scaled))
print(cm)
print('#'*50)
#-----------------------------
#Visulization Confusion Matrix
sns.heatmap(cm,annot=True,fmt='d',
            xticklabels=['Died','Survived'],
            yticklabels=['Died','Survived'],
            cmap='Blues')
plt.title('Confusion Matrix - Titanic')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()