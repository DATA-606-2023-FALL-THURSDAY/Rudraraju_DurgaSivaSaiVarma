# <p align ="center"> SMS & Email Spam Classifier </p>

##  Title and Author
- Project Title - **SMS & Email Spam prediction**
- Prepared for UMBC Data Science Master Degree Capstone by **Dr. Chaojie (Jay) Wang**
- Author - **Rudraraju Durga Siva Sai Varma**
- Author's GitHub profile - https://github.com/DATA-606-2023-FALL-THURSDAY/Rudraraju_DurgaSivaSaiVarma
- Author's LinkedIn profile - www.linkedin.com/in/saivarmarudraraju3157
- PowerPoint presentation file -
- YouTube video -

## 2. Background
- The objective of the Email Spam Classifier and SMS Spam Classifier projects is to develop machine learning models and algorithms that can automatically categorize emails and texts as spam (unwanted, unsolicited) or legitimate (non-spam) depending on their content and features. The main objective is to build a platform that classifies out undesirable messages, which will enhance email and messaging for users.
## 
- User Experience: Unwanted spam emails and SMS messages may annoy users and take up valuable time by clogging up inboxes. This project attempts to improve the user experience by effectively detecting and filtering out spam and ensuring that critical messages are not lost among irrelevant ones.
- Security and privacy: Spam emails can include unsafe content, phishing attempts, or fraudulent schemes. This project may contribute to increased security and safeguard consumers from potential dangers and scams by properly recognizing and blocking spam.
- Resource Efficiency: Spam messages consume computing, storage, and network resources. Particularly with large email and messaging systems, lowering the amount of spam can result in more effective resource management.
## 
- Which machine learning algorithms and techniques are most effective for spam classification in emails and SMS messages?
- What features or attributes of emails and SMS messages are most indicative of spam content?
- Are there any common patterns or keywords that distinguish spam messages from non-spam messages?

## 3. Data Description
- Data Sources: The dataset is obtained from Kaggle and is related to the SMS Spam Collection Dataset Collection of SMS messages tagged as spam or legitimate.
- Data Set Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download
- Data Size: The dataset is relatively small, with a size of approximately 503 KB.
- Data Description: The "spam.csv" dataset contains information related to SMS messages or emails, particularly focusing on spam classification. It is designed for machine learning tasks to predict whether a message is spam or not based on various features and attributes.
- Data Fields:
    v1 (Label): A binary classification label indicating whether a message is spam or not. Typically, it takes values like "ham" (for non-spam) and "spam" (for spam).
    v2 (Text): The text content of the SMS message or email.
- Dataset Size:
    Number of Rows (Samples): 5572
    Number of Columns (Features): 2
- Data Types:
    v1 (Label): Categorical (binary)
    v2 (Text): Text data
  
## 4. Exploratory Data Analysis (EDA)
 ### 4.1 Data Cleaning 
  #### 4.1.1 Changing the name for the required columns and dropping the unwanted columns
- In the DataFrame we have 5 columns they are v1, v2, Unnamed: 2, Unnamed: 3, and Unnamed: 4.  
- Unnamed: 2, Unnamed: 3, and Unnamed: 4 are the unwanted columns so they are deleted and the “v1” column is renamed to “label” and “v2” column are renamed to “text”.

#### 4.1.2 Checking & removing the duplicate rows from DataFrame
- Checking the duplicate rows in the DataFrame. It is observed that 403 duplicate rows are present in the DataFrame. After identifying the duplicate data, the very first value are kept and the rest duplicate values are been deleted.

 ### 4.2 visualization
- The number of Ham and the number of Spam messages are being counted. Then the percentage of Ham and Spam messages are represented in the Pie Chart
- **Count plot:** The below count plot represents the count vs label graph. From the plot, we can conclude that in the data that we have taken the count of Ham messages is above 4000 and the count of spam messages is below 1000. 

- **Pie Chart:** The pie chart represents the percentage of Ham and Spam messages that are present in the data. From the pie chart, we can conclude that in the data the percentage of Ham messages is 87.37% and the percentage of Spam messages is 12.63%.
  
  <img width="500" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Count%20of%20ham%20and%20spam%20messages.png">  <img width="400" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Pie%20chart%20of%20ham%20and%20spam%20messages.png">
 ### 4.3 Changing the categorical data 
- Changing the ham and spam categorical data to numerical data using the encoder
- The Ham and Spam messages are originally Categorical data then they are changed into Numeric data Ham is changed into 1 and Spam is changed into 0.

 ### 4.4 Importing the nltk
- NLTK stands for Natural Language Toolkit. It is used for working with human language data, particularly for tasks related to natural language processing (NLP). NLTK provides tools, resources, and libraries for a wide range of NLP tasks, including tokenization, stemming, tagging, parsing, semantic reasoning, and more.
- Created new columns from the text column using the NLTK libraries such as number of characters, number of words, number of sentences columns, and displayed the data frame.
<img width="708" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/NLTK-%20df%20of%20words%2C%20char%2C%20sentences.png">

- Plotted the graphs for Histogram of the Total Number of Characters by Label, Histogram of the Total Number of Words by Label,   Histogram of Total Number of Sentences by Label and Heatmap of number of characters, number of words, number of sentences columns

- **Histogram of the Total Number of Characters by Label:** This histogram represents the Count of the Total Number of characters by label. This means that it represents the count of the number of characters present in each message in the data. The Ham message is represented with blue color and the Spam message is represented with red. From this Histogram we can conclude that in the data the total number of characters is more in Ham messages when compared to that of Spam messages.
<img width="908" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Plot%20for%20the%20Number%20of%20Characters.png"> 

- **Histogram of the Total Number of Words by Label:** This histogram represents the Count of the Total Number of words by label. This means that it represents the count of the number of words that are present in each message in the data. The Ham message is represented with blue color and the Spam message is represented with red. From this Histogram we can conclude that in the data the total number of words is more in Ham messages when compared to that of Spam messages.
<img width="908" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Plot%20for%20the%20Number%20of%20Words.png"> 

- **Histogram of Total Number of Sentences by Label:** This histogram represents the Count of the Total Number of Sentences by label. This means that it represents the count of the number of Sentences present in each message in the data. The Ham message is represented with blue color and the Spam message is represented with red. From the Histogram we can conclude that in the data the total number of Sentences is more in Ham messages when compared to that of Spam messages.
<img width="908" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Plot%20for%20the%20Number%20of%20sentences.png">

- **Heatmap representing the correlation between the columns:** From the heatmap, we aim to see the correlation between the label which is the target, and the features which are total_num_of_charecters, total_num_of_words, and total_num_of_sentences. To make the model we will select one feature which has a high correlation with the label.  The higher the correlation the chances of the message being spam is more. In the Heatmap we can see that the correlation of Total_num_of_charecters and label is 0.38 which is the highest when compared to total_num_of_words, and total_num_of_sentences which are 0.26 and 0.27 respectively. From this, we can conclude that we will be using Total_num_of_charecters to make the model. 
<img width="708" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Heatmap%20of%20number%20of%20characters%2C%20number%20of%20words%2C%20number%20of%20sentences%20columns.png">

### 4.5 Processing the data
- NLTK allows you to tokenize the data which means it helps to break text into individual words or tokens, making it easier to analyze and manipulate text data. NLTK can be used in collaboration with machine learning techniques to build NLP models for tasks such as text classification, topic modeling and text generation. NLTK offers tools for parsing, allowing you to analyze the grammatical structure and meaning of sentences. With the help of “punkt” we can classify stopwords and non-alphanumeric characters which later can be removed if necessary. Additionally, it has capabilities for analyzing text's semantic meaning, such as semantic similarity tests and WordNet integration for word sense classification.
- Using the NLTK created the transformed_text function which gives the text as output, the transformed_text function will Convert to lowercase and tokenize the text and Remove non-alphanumeric characters, stopwords, and apply stemming to the text
<img width="908" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/NLKT%20-%20transformed_text.png">
  
### 4.6 Wordcloud
- Displaying the wordcloud Wordclous is the graphical representation of word frequency that give greater preference to words that appear more frequently in a source text. Using the wordcloud library I am creating two wordclouds one which shows spam words and the other which shows non-spam words.
- From the word cloud of spam messages, we can see that the frequency of words like text, free, u, and call are high in spam messages and that the frequency of words like go, u, come, got, call and love are high in Ham messages of the data.
- **Word cloud for the spam messages:**
<img width="508" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Word%20cloud%20for%20the%20spam%20messages.png">

- **Word cloud for the ham messages:**
<img width="508" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Word%20cloud%20for%20the%20ham%20messages.png">

 ### 4.7 Displaying the top 30 common words in the text
 - Created a list of spam and ham words from the data and found the Total spam words in the data and Total ham words in the data
 - **Plotted the Bar plot for the common words in the spam text:** The bar plot represents the Top 30 Frequent Words V/s Frequency of word in the spam data. We can clearly see that the frequency of the word “call” is highest in the spam messages which is higher than 300.
<img width="908" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Bar%20plot%20for%20the%20common%20words%20in%20the%20spam%20text.png">

 - **Bar plot for the common words in the ham text:** The bar plot represents the Top 30 Frequent Words V/s Frequency of word in the Ham data. We can clearly see that the frequency of the word “u” is highest in Ham messages which is Higher than 800.
<img width="908" alt="image" src="https://github.com/Rdssvarma/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Images/Bar%20plot%20for%20the%20common%20words%20in%20the%20ham%20text.png">

## 5. Model Training
 ### 5.1 Data Cleaning

 What models you will be using for predictive analytics?
- How will you train the models?
  - Train vs test split (80/20, 70/30, etc.)
  - Python packages to be used (scikit-learn, NLTK, spaCy, etc.)
  - The development environments (your laptop, Google CoLab, GitHub CodeSpaces, etc.)
- How will you measure and compare the performance of the models?



## 5. Model Training
### 5.1 Models Used for Machine Learning
   1. svc = SVC(kernel='sigmoid', gamma=1.0)
   2. knc = KNeighborsClassifier()
   3. mnb = MultinomialNB()
   4. dtc = DecisionTreeClassifier(max_depth=5)
   5. lrc = LogisticRegression(solver='liblinear', penalty='l1')
   6. rfc = RandomForestClassifier(n_estimators=50, random_state=2)
   7. abc = AdaBoostClassifier(n_estimators=50, random_state=2)
   8. bc = BaggingClassifier(n_estimators=50, random_state=2)
   9. etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
   10. gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
   11. xgb = XGBClassifier(n_estimators=50,random_state=2)
       
 ### 5.2 Process of Training The Models
 #### 5.2.1.Train-Test Split
- We divided the data into an 80/20 train-test split, using 80% of the data for training the models and 20% for testing. With the use of this split, it is possible to assess how effectively the models generalize to differentiate the spam and ham text.
  
 #### 5.2.2 Data Preprocessing
- **Vectorization:** TfidfVectorizer is a text feature extraction technique that is frequently used in text mining and natural language processing (NLP). They can be used to transform text data into numerical representations that machine-learning models can use. 

 #### 5.2.3 Python Packages
For a variety of tasks throughout the project, we used the Python packages and libraries listed below:
- scikit-learn: For building, training, and evaluating machine learning models.
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- matplotlib and seaborn: For data visualization.

 #### 5.2.4 Development Environment
Kaggle served as the project's development environment since it offers a Jupyter Notebook environment with access to crucial tools and resources for data science and machine learning. Kaggle is appropriate for data science projects since it enables seamless collaboration, code exchange, and simple access to datasets.

 ### 5.3 Measuring and Comparing Model Performance
- Accuracy: This statistic assesses how accurately the model's predictions were made in general. It is the proportion of accurate forecasts to all predictions.
- Precision: How much of the model's favorable predictions were accurate is known as precision. In order to compute it, divide the total number of true positives by the total number of false positives, or TP / (TP + FP).
- Confusion_matrix: A confusion matrix gives a thorough description of the right and wrong guesses. True positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) are a few of the measures it offers.

 ### 5.4 Result and Interpretations
**1. GaussianNB:**   
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Rudraraju_DurgaSivaSaiVarma/assets/143572017/3b347534-e8d1-4530-b10f-b5bde23ddf03)

- True Positives (TP): 113 - The model correctly predicted that 113 messages are Spam messages.
- True Negatives (TN): 783 - The model correctly predicted that 783 messages are Ham messages.
- False Positives (FP): 113 - The model incorrectly predicted that 113 messages were Spam messages when they were Ham messages.
- False Negatives (FN): 25 - The model incorrectly predicted that 25 messages were Ham messages when they were Spam messages.

**Interpretation:** 
- Accuracy:
- Precision:

**2. MultinomialNB:**

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Rudraraju_DurgaSivaSaiVarma/assets/143572017/ca8fd2c6-6e0f-4af8-98ea-4d1b7d908978)

- True Positives (TP): 110 - The model correctly predicted that 110 messages are Spam messages.
- True Negatives (TN): 896 - The model correctly predicted that 896 messages are Ham messages.
- False Positives (FP): 0 - The model incorrectly predicted that 0 messages were Spam messages when they were Ham messages.
- False Negatives (FN): 28 - The model incorrectly predicted that 28 messages were Ham messages when they were Spam messages.

**Interpretation:** 
- Accuracy:
- Precision:
  
**3. BernoulliNB:**

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Rudraraju_DurgaSivaSaiVarma/assets/143572017/ec95b720-f471-464b-898d-8c8d3b998529)

- True Positives (TP): 120 - The model correctly predicted that 120 messages are Spam messages.
- True Negatives (TN): 895 - The model correctly predicted that 895 messages are Ham messages.
- False Positives (FP): 1 - The model incorrectly predicted that 1 message was a Spam message when they were a Ham message.
- False Negatives (FN): 18 - The model incorrectly predicted that 18 messages were Ham messages when they were Spam messages.

**Interpretation:** 
- Accuracy:
- Precision:

**Comparing All The Models:**

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Rudraraju_DurgaSivaSaiVarma/assets/143572017/122db7e2-3fe9-4185-9a9f-026bdce1876e)

I have trained the models with the data and found out the precision and accuracy of each model. After testing all the models I can see NB model is working well 

# 6. Application of Trained Model
In this project, we created a user-friendly web application using Streamlit. This will make our trained machine-learning predictive model available publicly. With this web application customers will be able to interact with our model which will tell them if their message is spam or not spam. The main reason for choosing Streamlit is because it is an open-source Python library that will quickly turn data science and machine learning projects into web applications. 

## 6.1 Web Application Overview
This web application will give the user a simple user-friendly interface where a user will find a message box where the user will be able to give a message and check if it a spam or not. 
**Link - https://sms-email-classifier-vprf.onrender.com/**

## 6.2 Features of the Web Application
1.	Input Fields: The user will find the message box where he will be able to give the message they want to check if it a spam or not spam. 
2.	Predictive Power: Our machine-learning model was accurately trained on a large dataset. So, when a user gives a message, it will predict accurately. 
3.	User-Friendly Feedback: Users receive simple responses where the model will predict if the message is spam or not spam. 
4.	Accessibility: Its web application is made to be accessed from several platforms and on different devices, making it easy for users to access on desktop and mobile platforms.

## 6.3 How to Use the Web Application
Using the application is a straightforward process:
1.	You can use any web browser to access the application.
2.	Next, give the message you want to check in the message box. 
3.	To predict the message just click on the predict button. 
4.	Then the model will predict if the given message is spam or not

## 6.4 Why Streamlit
Streamlit is an open-source Python tool that makes web applications out of data science and machine learning projects very rapidly. For data scientists and machine learning researchers whose primary areas of expertise are developing prototypes, testing models, and working with data in Python, this framework is excellent. Streamlit is rapid, efficient, adaptable, and simple to use.

## 6.5 Screenshots

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Rudraraju_DurgaSivaSaiVarma/assets/143572017/436c8174-4546-4498-a0ca-b3efff381754)

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Rudraraju_DurgaSivaSaiVarma/assets/143572017/392e9aad-f78f-4286-94f3-b041e34d489d)



