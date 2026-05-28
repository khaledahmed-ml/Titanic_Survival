import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
#------------------------------------
# Load and clean data
df = pd.read_csv("Titanic_train.csv")
df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Embarked', 'Ticket'])
# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
# Encode categorical features
df['Sex'] = df['Sex'].astype("category").cat.codes
df = df.dropna()
# Split and scale data
X = df.drop(columns='Survived')
Y = df['Survived']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# Train model
clf=SVC(max_iter=10000,kernel='rbf',C=5,gamma='auto',probability=True)
clf.fit(X_train_scaled,Y_train)
#------------------
#Accuracy
print(f"The Accuracy of Train: {clf.score(X_train_scaled,Y_train)}")
print(f"The Accuracy of Test: {clf.score(X_test_scaled,Y_test)}")
#------------------
will_you_survive = pd.DataFrame([{
    'Pclass': 1,
    'Sex':    0,
    'Age':    25,
    'SibSp':  0,
    'Parch':  0,
    'Fare':  70 
}])
will_you_survive_scaled = scaler.transform(will_you_survive)
prediction  = clf.predict(will_you_survive_scaled)
probability = clf.predict_proba(will_you_survive_scaled)

print(f"The Prediction         : {'Survived' if prediction[0] == 1 else 'Not Survived'}")
print(f"Probability of survival: {probability[0][1]:.2%}")
print(f"Possibility of death   : {probability[0][0]:.2%}")
print('#'*50)
#------------------------
#classification_report
Y_pred = clf.predict(X_test_scaled)
print(classification_report(Y_test, Y_pred, target_names=['Died', 'Survived']))
#------------------------
#Confusion Matrix
cm=confusion_matrix(Y_test,clf.predict(X_test_scaled))
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
