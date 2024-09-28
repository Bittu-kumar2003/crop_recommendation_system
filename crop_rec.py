#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact


# In[2]:


data=pd.read_csv("crop.csv")


# In[3]:


print("Shape of the Dataset:",data.shape)


# In[4]:


#lets check the head of the dataset
data.head()


# In[5]:


#lets check if there is any missing value present in the dataset
data.isnull().sum()


# In[6]:


#lets check the crops present in the dataset
data["label"].value_counts()


# In[7]:


#lets check the summary for all the crops
print("Average Ratio of Nitrogen in the soil :{0:.2f}".format(data['N'].mean()))
print("Average Ratio of  Phosphorous in the soil :{0:.2f}".format(data['P'].mean()))
print("Average Ratio of  Potassium in the soil :{0:.2f}".format(data['K'].mean()))
print("Average Ratio of  Tempature in Celsius is :{0:.2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in %:{0:.2f}".format(data['humidity'].mean()))
print("Average PH Value of the soil:{0:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm:{0:.2f}".format(data['rainfall'].mean()))


# In[8]:


# Lets check the Summary Statistics for each of the Crops
def summary(crops=list(data['label'].value_counts().index)):
    for crop in crops:
        x = data[data['label'] == crop]
        print("-----------------------")
        print(f"Statistics for {crop.capitalize()}")
        print("Minimum Nitrogen required :", x['N'].min())
        print("Average Nitrogen required :", x['N'].mean())
        print("Maxmium Nitrogen required :", x['N'].max())
        print("-----------------------")
        print("Statistics for Phosphorous")
        print("Minimum Phosphorous required :", x['P'].min())
        print("Average Phosphorous required :", x['P'].mean())
        print("Maxmium Phosphorous required :", x['P'].max())
        print("-----------------------")
        print("Statistics for Potassium")
        print("Minimum Potassium required :", x['K'].min())
        print("Average Potassium required :", x['K'].mean())
        print("Maxmium Potassium required :", x['K'].max())
        print("-----------------------")
        print("Statistics for Tempature")
        print("Minimum Tempature in Celsius required :{0:.2f}".format(x['temperature'].min()))
        print("Average Tempature in Celsius required :{0:.2f}".format(x['temperature'].mean()))
        print("Maxmium Tempature in Celsius required :{0:.2f}".format(x['temperature'].max()))
        print("-----------------------")
        print("Statistics for Humidity")
        print("Minimum Humidity required :{0:.2f}".format(x['humidity'].min()))
        print("Average Humidity required :{0:.2f}".format(x['humidity'].mean()))
        print("Maxmium Humidity required :{0:.2f}".format(x['humidity'].max()))
        print("-----------------------")
        print("Statistics for PH")
        print("Minimum PH required :{0:.2f}".format(x['ph'].min()))
        print("Average PH required :{0:.2f}".format(x['ph'].mean()))
        print("Maxmium PH required :{0:.2f}".format(x['ph'].max()))
        print("-----------------------")
        print("Statistics for Rainfall")
        print("Minimum Rainfall required :{0:.2f}".format(x['rainfall'].min()))
        print("Average Rainfall required :{0:.2f}".format(x['rainfall'].mean()))
        print("Maxmium Rainfall required :{0:.2f}".format(x['rainfall'].max()))

user_value = input("Press the number corresponding to the item:\n1. Rice\n2. Maize\n3. Jute\n4. Cotton\n5. Coconut\n6. Papaya\n7. Orange\n8. Apple\n9. Muskmelon\n10. Watermelon\n11. Grapes\n12. Mango\n13. Banana\n14. Pomegranate\n15. Lentil\n16. Blackgram\n17. Mungbean\n18. Mothbeans\n19. Pigeonpeas\n20. Kidneybeans\n21. Chickpea\n22. Coffee\n")
user_value = int(user_value)
items = ["rice", "maize", "jute", "cotton", "coconut", "papaya", "orange", "apple", "muskmelon", "watermelon", "grapes", "mango", "banana", "pomegranate", "lentil", "blackgram", "mungbean", "mothbeans", "pigeonpeas", "kidneybeans", "chickpea", "coffee"]

if 1 <= user_value <= len(items):
    selected_item = items[user_value - 1]
    print(f"You selected: {selected_item}")
    summary(crops=[selected_item])
else:
    print("Invalid selection. Please choose a number within the specified range.")


# In[9]:


@interact
def compare(conditions=['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']):
    print("Average Value for", conditions, "is {0:.2f}".format(data[conditions].mean()))
    print("---- ---")
    print("Rice: {0:.2f}".format(data[data['label'] == 'rice'][conditions].mean()))
    print("Black Grams: {0:.2f}".format(data[data['label'] == 'blackgram'][conditions].mean()))
    print("Banana: {0:.2f}".format(data[data['label'] == 'banana'][conditions].mean()))
    print("Jute: {0:.2f}".format(data[data['label'] == 'jute'][conditions].mean()))
    print("Coconut: {0:.2f}".format(data[data['label'] == 'coconut'][conditions].mean()))
    print("Apple: {0:.2f}".format(data[data['label'] == 'apple'][conditions].mean()))
    print("Papaya: {0:.2f}".format(data[data['label'] == 'papaya'][conditions].mean()))
    print("Muskmelon: {0:.2f}".format(data[data['label'] == 'muskmelon'][conditions].mean()))
    print("Grapes: {0:.2f}".format(data[data['label'] == 'grapes'][conditions].mean()))
    print("Watermelon: {0:.2f}".format(data[data['label'] == 'watermelon'][conditions].mean()))
    print("Kidney Beans: {0:.2f}".format(data[data['label'] == 'kidneybeans'][conditions].mean()))
    print("Mung Beans: {0:.2f}".format(data[data['label'] == 'mungbean'][conditions].mean()))
    print("Oranges: {0:.2f}".format(data[data['label'] == 'orange'][conditions].mean()))
    print("Chick Peas: {0:.2f}".format(data[data['label'] == 'chickpea'][conditions].mean()))
    print("Lentils: {0:.2f}".format(data[data['label'] == 'lentil'][conditions].mean()))
    print("Cotton: {0:.2f}".format(data[data['label'] == 'cotton'][conditions].mean()))
    print("Maize: {0:.2f}".format(data[data['label'] == 'maize'][conditions].mean()))
    print("Moth Beans: {0:.2f}".format(data[data['label'] == 'mothbeans'][conditions].mean()))
    print("Pigeon Peas: {0:.2f}".format(data[data['label'] == 'pigeonpeas'][conditions].mean()))
    print("Mango: {0:.2f}".format(data[data['label'] == 'mango'][conditions].mean()))
    print("Pomegranate: {0:.2f}".format(data[data['label'] == 'pomegranate'][conditions].mean()))
    print("Coffee: {0:.2f}".format(data[data['label'] == 'coffee'][conditions].mean()))


# In[10]:


@interact
def compare (conditions = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']):
    print("Crops which require greater than average", conditions, '\n') 
    print(data[data[conditions] > data [conditions].mean()]['label']. unique())
    print("--------------")
    print("Crops which require less than average", conditions, '\n') 
    print(data[data[conditions] <= data[conditions].mean()]['label'].unique())


# In[11]:


# Let's find out some Interesting Facts
print("Some Interesting Patterns")
print("-----------------------------------")
# Crops which require very High Ratio of Nitrogen Content in Soil
print("Crops which require very High Ratio of Nitrogen Content in Soil:", data[data['N'] > 120]['label'].unique())

# Crops which require very High Ratio of Phosphorous Content in Soil
print("Crops which require very High Ratio of Phosphorous Content in Soil:", data[data['P'] > 100]['label'].unique())

# Crops which require very High Ratio of Potassium Content in Soil
print("Crops which require very High Ratio of Potassium Content in Soil:", data[data['K'] > 200]['label'].unique())

# Crops which require very High Rainfall
print("Crops which require very High Rainfall:", data[data['rainfall'] > 200]['label'].unique())

# Crops which require very Low Temperature
print("Crops which require very Low Temperature:", data[data['temperature'] < 10]['label'].unique())

# Crops which require very High Temperature
print("Crops which require very High Temperature:", data[data['temperature'] > 40]['label'].unique())

# Crops which require very Low Humidity
print("Crops which require very Low Humidity:", data[data['humidity'] < 20]['label'].unique())

# Crops which require very Low pH
print("Crops which require very Low pH:", data[data['ph'] < 4]['label'].unique())

# Crops which require very High pH
print("Crops which require very High pH:", data[data['ph'] > 9]['label'].unique())


# In[12]:


# Identify crops for different seasons
print("Summer Crops:")
summer_crops = data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique()
print(summer_crops)
print("-----------------------------------")
print("\nWinter Crops:")
winter_crops = data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique()
print(winter_crops)
print("-----------------------------------")
print("\nRainy Crops:")
rainy_crops = data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique()
print(rainy_crops)
print("-----------------------------------")


# In[13]:


from sklearn.cluster import KMeans

# Removing the Labels column
x = data.drop(['label'], axis=1)

# Selecting all the values of the data
x = x.values

# Checking the shape
print(x.shape)


# In[14]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = (10, 4)
WCSS = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    WCSS.append(km.inertia_)

# Plot the results
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method', fontsize=20)
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[15]:


from sklearn.cluster import KMeans
import pandas as pd

# Implement K Means algorithm
km = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

# Find out the Results
a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis=1)
z = z.rename(columns={0: 'cluster'})

# Display the Clusters of each Crops
print("Results After Applying the K Means Clustering Analysis\n")
print("Crops in Second Cluster:", z[z['cluster'] == 0]['label'].unique())
print("------------------------")
print("Crops in First Cluster:", z[z['cluster'] == 1]['label'].unique())
print("------------------------")
print("Crops in Third Cluster:", z[z['cluster'] == 2]['label'].unique())
print("------------------------")
print("Crops in Fourth Cluster:", z[z['cluster'] == 3]['label'].unique())


# In[16]:


# lets splists  the dataset for Predictive Modelling
y = data['label']
x = data.drop(['label'], axis=1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[17]:


# lets  create  Training and Testing set of Validation  of Results
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("The Shape of x train:", x_train.shape)
print("The Shape of x test:", x_test.shape)
print("The Shape of y train:", y_train.shape)
print("The Shape of y test:", y_test.shape)


# In[18]:


# from sklearn.linear_model import LogisticRegression

# # Create a Logistic Regression model
# model = LogisticRegression()

# # Fit the model on the training data
# model.fit(x_train, y_train)

# # Make predictions on the testing data
# y_pred = model.predict(x_test)
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# Assuming you have loaded your data into x and y
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(x_test)


# In[19]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your data and labels (X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using a heatmap
plt.rcParams['figure.figsize'] = (10, 10)
sns.heatmap(cm, annot=True, cmap='Wistia', fmt='g')
plt.title('Confusion Matrix for Logistic Regression', fontsize=15)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[20]:


# # lets print the  Classification Report also
# cr = classification_report(y_test, y_pred)
# print(cr)
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming you have loaded your data into x and y
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(x_test)

# Print the classification report
cr = classification_report(y_test, y_pred)
print(cr)


# In[21]:


#lets check the Head of the  Dataset
data.head()


# In[23]:


import numpy as np

# Assuming you have your features in a NumPy array
features = np.array([[40, 40, 40, 20, 80, 7, 200]])

# Replace 'model' with the variable name of your trained model
prediction = model.predict(features)
print("The Suggested Crop for Given Climatic Condition is", prediction)


# In[ ]:




