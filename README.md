# Business-Analytics-CAPSTONE-Project
Amazon Review Rating Prediction

Project Objective

The goal of this project was to build and evaluate a model to predict the numerical star rating (1 to 5 stars) of an Amazon product review based solely on its text content (title and body). This was treated as a regression task to predict a continuous numerical value.

Dataset

The analysis was performed on a processed Amazon electronics review dataset, processed_electronics_reviews_openrouter_threaded.jsonl. A subset of 10,000 reviews was used for the Feature Extraction approach, while a smaller subset of 1,000 training and 200 test reviews was used for the Fine-tuning approach to ensure computational efficiency during model comparison.

Methodology

Two distinct approaches were implemented and compared to solve this regression problem:

Approach 1: 
Fine-Tuning a Pre-trained Language Model (PLM)
This method involved adapting a pre-trained transformer model to perform a specific regression task.

Model: A DistilBERT model (distilbert-base-uncased) was used. It was modified with a linear regression head to output a single numerical value.

Training: A custom PyTorch training loop was implemented for a more controlled training process. The model was fine-tuned for three epochs.

Metrics: Performance was measured using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R 
2
 ).



Approach 2: Feature Extraction + Traditional Machine Learning
This approach separated the task into two steps: converting text into numerical features and then training a traditional machine learning model on those features.

Feature Extraction: Two different feature extraction methods were tested:
TF-IDF: Text was vectorized using a TF-IDF vectorizer with a max_features limit of 5000.
SentenceTransformer: A pre-trained all-MiniLM-L6-v2 model was used to generate dense vector embeddings for each review.

Regression Models: A variety of scikit-learn regression models were tested on the extracted features, including:
Linear Models (Linear, Ridge, Lasso, ElasticNet)

Support Vector Machines (Linear SVR, SVR)

Tree-based Models (Decision Tree, Random Forest, Gradient Boosting, XGBoost)

Other models (K-Nearest Neighbors, AdaBoost)

RESULTS AND ANALYSIS:

Model Comparison

The fine-tuning approach generally demonstrated superior predictive performance, while the feature extraction approach was significantly more computationally efficient.


To help you upload your project to GitHub and provide a detailed README file, I have analyzed the uploaded Jupyter Notebook file. It contains a comprehensive comparison of two different approaches for predicting Amazon review ratings, along with code, results, and a concluding summary.

I will provide you with a structured guide for uploading the project and a detailed README that accurately reflects your work.

README File for Your Project
Here is a detailed README file that you can use. It is designed to be professional, thorough, and suitable for a capstone project. It summarizes your findings, methodology, and the results from both approaches.

CA9.1: Creating Predictions from Text
Project Objective
The goal of this project was to build and evaluate a model to predict the numerical star rating (1 to 5 stars) of an Amazon product review based solely on its text content (title and body). This was treated as a regression task to predict a continuous numerical value.

Dataset
The analysis was performed on a processed Amazon electronics review dataset, processed_electronics_reviews_openrouter_threaded.jsonl. A subset of 10,000 reviews was used for the Feature Extraction approach, while a smaller subset of 1,000 training and 200 test reviews was used for the Fine-tuning approach to ensure computational efficiency during model comparison.

Methodology
Two distinct approaches were implemented and compared to solve this regression problem:

Approach 1: Fine-Tuning a Pre-trained Language Model (PLM)
This method involved adapting a pre-trained transformer model to perform a specific regression task.

Model: A DistilBERT model (distilbert-base-uncased) was used. It was modified with a linear regression head to output a single numerical value.

Training: A custom PyTorch training loop was implemented for a more controlled training process. The model was fine-tuned for three epochs.

Metrics: Performance was measured using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R 
2
 ).

Approach 2: Feature Extraction + Traditional Machine Learning
This approach separated the task into two steps: converting text into numerical features and then training a traditional machine learning model on those features.

Feature Extraction: Two different feature extraction methods were tested:

TF-IDF: Text was vectorized using a TF-IDF vectorizer with a max_features limit of 5000.

SentenceTransformer: A pre-trained all-MiniLM-L6-v2 model was used to generate dense vector embeddings for each review.

Regression Models: A variety of scikit-learn regression models were tested on the extracted features, including:

Linear Models (Linear, Ridge, Lasso, ElasticNet)

Support Vector Machines (Linear SVR, SVR)

Tree-based Models (Decision Tree, Random Forest, Gradient Boosting, XGBoost)

Other models (K-Nearest Neighbors, AdaBoost)

Results and Analysis
Model Comparison
The fine-tuning approach generally demonstrated superior predictive performance, while the feature extraction approach was significantly more computationally efficient.


Key Findings:

The fine-tuning approach with DistilBERT achieved the best overall accuracy, with an average prediction error of just 0.26 stars. Its higher R 
2
  value indicates a superior ability to capture the complex relationship between review text and ratings.

The feature extraction approach, particularly with TF-IDF features and a Ridge Regression model, provided a strong baseline. It predicted ratings with an average error of 0.33 stars, which is a respectable result for a much faster and simpler model.

Both approaches struggle with short reviews (e.g., "Na - Na...") and those where the sentiment in the text contradicts the numerical rating.

The choice of approach depends on the trade-off between prediction accuracy, training time, and computational resources. Fine-tuning is ideal for applications where accuracy is the top priority, while feature extraction is excellent for rapid prototyping or resource-constrained environments.

Files in this Repository:

My Work on Project: This is the main Jupyter Notebook containing all the code for data loading, model implementation (both fine-tuning and feature extraction), and the detailed analysis and visualizations.

Project Description: The dataset used in this project.

Tools and Libraries Used:

pandas and numpy: These are foundational libraries for data manipulation and numerical operations. pandas is used to load and clean the JSONL dataset, and numpy is used for handling numerical arrays, such as the target ratings and predictions.

matplotlib.pyplot and seaborn: These libraries are used for data visualization. You use them to plot the distribution of ratings, compare model performance, and visualize the predictions versus the actual ratings.

scikit-learn: A versatile machine learning library that provides a consistent interface for many supervised and unsupervised learning algorithms. You use it for:

Data Splitting: train_test_split is used to divide the dataset into training and testing sets.

Feature Extraction: TfidfVectorizer is a key component for the traditional machine learning approach.

Preprocessing: StandardScaler is used to normalize data before it's fed into the models.

Regression Models: You compare a variety of models, including LinearRegression, Ridge, Lasso, ElasticNet, SVR, and ensemble models like GradientBoostingRegressor, RandomForestRegressor, and XGBRegressor.

Evaluation: Metrics like mean_absolute_error, mean_squared_error, and r2_score are used to evaluate model performance.

transformers (from Hugging Face): This library is a cornerstone of modern NLP. You use it for the fine-tuning approach:

Tokenization: AutoTokenizer is used to convert text into numerical tokens that a transformer model can understand.

Model Loading: AutoModel is used to load pre-trained models like DistilBERT, RoBERTa, and BERT.

sentence-transformers: A specialized framework built on top of transformers for generating fixed-size sentence embeddings. You use this to create high-quality feature vectors for the traditional machine learning models.

torch: A deep learning framework used as the backend for the transformers library, particularly for building and training your custom fine-tuned model.

tqdm: This library is used to display progress bars during long-running tasks, such as data loading and generating embeddings.


