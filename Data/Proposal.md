# <p align ="center"> SMS & Email Spam Classifier </p>

##  Title and Author
- Project Title - ** SMS & Email Spam prediction **
- Prepared for UMBC Data Science Master Degree Capstone by **Dr. Chaojie (Jay) Wang**
- Author - **Rudraraju Durga Siva Sai Varma**
- Author's GitHub profile - https://github.com/DATA-606-2023-FALL-THURSDAY/Rudraraju_DurgaSivaSaiVarma
- Author's LinkedIn profile - www.linkedin.com/in/saivarmarudraraju3157
- PowerPoint presentation file -
- YouTube video -

##  Background
- The objective of the Email Spam Classifier and SMS Spam Classifier projects is to develop machine learning models and algorithms that can automatically categorize incoming emails and texts as spam (unwanted, unsolicited) or legitimate (non-spam) depending on their content and features. The main objective is to build a platform that filters out undesirable messages, which will enhance email and messaging for users.
## 
- User Experience: Unwanted spam emails and SMS messages may annoy users and take up valuable time by clogging up inboxes. This project attempts to improve the user experience by effectively detecting and filtering out spam and ensuring that critical messages are not lost among irrelevant ones.
- Security and privacy: Spam emails can include unsafe content, phishing attempts, or fraudulent schemes. This project may contribute to increased security and safeguard consumers from potential dangers and scams by properly recognizing and blocking spam.
- Resource Efficiency: Spam messages consume computing, storage, and network resources. Particularly with large email and messaging systems, lowering the amount of spam can result in more effective resource management.
## 
- Which machine learning algorithms and techniques are most effective for spam classification in emails and SMS messages?
- What features or attributes of emails and SMS messages are most indicative of spam content?
- How can the project's results be integrated into existing email and messaging platforms for practical use?
-

##  Data Description
- Data Sources: The dataset is obtained from Kaggle and is related to the SMS Spam Collection Dataset Collection of SMS messages tagged as spam or legitimate.
- Data Set Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download
- Data Size: The dataset is relatively small, with a size of approximately 503 KB.
- Data Shape: The dataset contains 5,572 rows and 5 columns.
- Data Dictionary:
  
    1.	ID: ID number of customers (Integer).
    2.	Warehouse_block: Warehouse block categorization (Categorical - A, B, C, D, E).
    3.	Mode_of_Shipment: Mode of shipment (Categorical - Ship, Flight, Road).
    4.	Customer_care_calls: The number of calls made for inquiries about the shipment (Integer).
    5.	Customer_rating: Customer rating (Integer, 1 to 5, with 1 being the lowest and 5 being the highest).
    6.	Cost_of_the_Product: Cost of the product in US Dollars (Integer).
    7.	Prior_purchases: The number of prior purchases made by the customer (Integer).
    8.	Product_importance: Importance categorization of the product (Categorical - Low, Medium, High).
    9.	Gender: Gender of the customer (Categorical - Male, Female).
    10.	Discount_offered: Discount offered on the specific product (Integer).
    11.	Weight_in_gms: Weight of the product in grams (Integer).
    12.	Reached.on.Time_Y.N: Target variable, where 1 indicates that the product has NOT reached on time, and 0 indicates it has reached on time (Binary classification).
Target Variable/Label:
The "Reached.on.Time_Y.N" variable, which denotes whether the product arrived on time or not, is the target variable for the machine learning models. It has the binary values of 1 (Not on time) and 0 (On time), and it is a variable.

