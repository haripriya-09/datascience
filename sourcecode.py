from google.colab import files
uploaded = files.upload()
import pandas as pd

# Assuming the uploaded file is 'hari.nm.csv' based on the global variables
# The 'latin-1' encoding is used as a possible solution.
# If this still gives errors, you might need to try other encodings like 'ISO-8859-1', 'cp1252', etc.
# You need to experiment with different encodings until the error is resolved.
df = pd.read_csv("hari.nm.csv", encoding='latin-1')
df.head()
df.info()
df.describe()
df['label'].value_counts()print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Drop duplicates if any
df = df.drop_duplicates()

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='label')
plt.title("Distribution of Fake vs Real News")
plt.show()
X = df['Text']
y = df['label']
# Convert 'label' to numerical: Fake = 0, Real = 1
df['label'] = df['label'].map({'Fake': 0, 'Real': 1})
y = df['label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Classifier
model = PassiveAggressiveClassifier()
model.fit(X_train_vec, y_train)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Classifier
model = PassiveAggressiveClassifier()
model.fit(X_train_vec, y_train)
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
sample_news = ["The government passed a new healthcare reform bill."]
sample_vec = vectorizer.transform(sample_news)
pred = model.predict(sample_vec)
print("Prediction:", "Real" if pred[0] == 1 else "Fake")
!pip install gradio
def predict_news(text):
    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]
    return "Real News 📰" if result == 1 else "Fake News ⚠️"
import gradio as gr

interface = gr.Interface(fn=predict_news,
                         inputs="text",
                         outputs="text",
                         title="🕵️‍♂️ Fake News Detector",
                         description="Enter a news headline or article to check if it's Real or Fake.")
interface.launch()
