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

- **Heatmap of number of characters, number of words, number of sentences columns:** From the heatmap, we aim to see the correlation between the label which is the target, and the features which are total_num_of_charecters, total_num_of_words, and total_num_of_sentences. To make the model we will select one feature which has a high correlation with the label.  The higher the correlation the chances of the message being spam is more. In the Heatmap we can see that the correlation of Total_num_of_charecters and label is 0.38 which is the highest when compared to total_num_of_words, and total_num_of_sentences which are 0.26 and 0.27 respectively. From this, we can conclude that we will be using Total_num_of_charecters to make the model. 
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

