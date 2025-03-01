import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gdown
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
from google.colab import drive

def download_data(file_id, destination):
  file_id = '14Oqmrcf2miZQjFHQd0r_n9Xva7fI-er1'
  gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet = False)
  download_data(file_id, destination)
for _ in range(1):
  if not os.path.exists('Determine BRI.csv'):
    download_data('14Oqmrcf2miZQjFHQd0r_n9Xva7fI-er1', 'Determine BRI.csv')
  else:
    print('File already exists')
data = pd.read_csv('Determine BRI.csv')  # Replace with your actual dataset path

# Define features (X) and target (y)
X = data.drop('Obesity Level', axis=1).values  # Replace 'obesity_level' with your actual target column name
y = data['Obesity Level'].values
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Create an SVM classifier
svm_classifier = svm.SVC(kernel='poly', C=10)

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))
# prompt: scatter diagram, confuse matrix and other graphs for the csv file
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(list(set(y))), yticklabels=sorted(list(set(y))))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Scatter Diagram (Example with first two features)
plt.figure(figsize=(8, 6))
le = LabelEncoder()
y_pred = le.fit_transform(y_pred)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Diagram of Predictions')
plt.colorbar(label='Predicted Class')
plt.show()


# Other Graphs (Example: Histogram of Target Variable)
plt.figure(figsize=(8, 6))
plt.hist(y, bins=len(set(y)), alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Obesity Level')
plt.ylabel('Frequency')
plt.title('Distribution of Obesity Level')
plt.show()

# Pair Plot (for visualizing relationships between multiple features)
sns.pairplot(data, hue='Obesity Level')
plt.show()

drive.mount('content/drive')
os.chdir('/content/drive/My Drive/YOUR_PATH') #Specify path of drive to sav.pkl file

model_filename = 'svm_model.pkl'
joblib.dump(svm_classifier, model_filename)
print("Model saved to", model_filename) #Displays the save name of model

#Now, check the drive where it is mounted and directory is listed. There, you can find the .pkl file

