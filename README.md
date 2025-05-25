# Legal Document Classifier

### Objective

This project aims to build a Natural Language Processing (NLP) pipeline to classify legal case reports into specific areas of law. The model is trained on a dataset of case reports to identify and categorize them based on their content.

To reproduce:

| Step                        | Command                                                 |
|-----------------------------|---------------------------------------------------------|
| Create virtual env          | `python -m venv venv`                                   |
| Activate virtual env (Win)  | `venv\Scripts\activate`                                 |
| Activate virtual env (Mac/Linux) | `source venv/bin/activate`                         |
| Install dependencies        | `pip install -r requirements.txt`                       |
| Run Streamlit app           | `streamlit run app/streamlit_app.py`                    |
| Test Flask API Endpoint     | `python test.py`                                        |
| Deactivate virtual env      | `deactivate`                                            |

The `LegalDocumentClassifier.ipynb` file contains the code for building the classifier model. The `app` folder contains the streamlit app and flask API endpoint and unittest. The model and tokenizer are in the 'model' folder.

### Dataset

The dataset used 200 rows and the following columns:

- `case_title`: Title of the legal case
- `suitno`: Suit number or case reference
- `introduction`: Brief introduction 
- `facts`: Summary of the facts of the case
- `issues`: Legal issues considered in the case
- `decision`: The decision or judgment
- `full_report`: The full body of the legal judgment

### Data Loading and Initial Exploration

This involved loading the dataset and performing initial checks for duplicates and missing values. There was only one row with missing value so it was dropped.

### Data Wrangling and Preprocessing

The Case_title, suitno and decision columns were dropped as they didn't contain useful information for the classification. From visual exploration, the `introduction` column was identified as to be the best source for extracting the target variable, which is the area of law. It was observed that most of the text in the column follwed a pattern where the area of law was explicitly stated. 

For example: "This appeal borders on Civil Procedure." "This appeal borders on Criminal Law and Procedure". "This is a ruling on an Application seeking Leave of Court..."

It was observed that many of the records in the `introduction` column followed this pattern. A split of the sentence after the word "on" (specifically the last "on" present in the sentence), and stripping of extra white space and full stops, yielded the area of law as stand-alone (e.g. "Civil Procedure") for most rows. However, a good number of the column still had many words which were too generalized to be used as area of law (e.g. "propriety of requirement of consent of the A.G. of the Federation in a garnishee proceedings against the CBN"). Further action was needed. 

A value count of the extracted "after_last_on" column showed that certain areas of law appeared more frequently. The top 5 most frequent areas were then used as labels with "Other" capturing any other that do not fall under the 5 areas. To more accurately assign labels to the rows that had too many words, the all-MiniLM-L6-v2 sentence transformer was used to check how similar the words in the extracted "after_last_on" column were to the six areas of law (5 plus Other). The cosine similarity metric was used for this and each row containing long words was mapped to their corresponding area of law. This formed the label column.

The text in the `full_report`, `introduction`, `facts`, and `issues` was then cleaned by handling non-breaking spaces, whitespace, punctuation, special characters, non-alphabetic characters, stopwords, and converting text to lowercase. 

### Tokenization and Vectorization

The cleaned feature columns were tokenized and converted into numerical representations using the keras Tokenizer. The categorical labels (area of law) were also encoded into numerical format.

### Model Building, Tuning and Evaluation 

A Decision Tree Classifier was trained on the preprocessed data and its performance was evaluated using a classification report and confusion matrix. The accuracy of 97% and values of the metrics (precision, recall, f1 score) showed that the model generalized well on the five core labels and accurately predicted them. GridSearchCV was used to find the best hyperparameters for the Decision Tree Classifier for a maximum performance. The trained model and tokenizer were saved to be implemented in an app.

### Implementation

A Flask API endpoint was created where a new case report can be submitted to return a predicted area of law.
Also, check out the Streamlit app here and upload case reports to get the predicted area of law.

### Recommendations and Future Improvements

- Train on a larger and more diverse dataset for improved generalization.
- Implement more sophisticated label extraction methods from the introduction.

