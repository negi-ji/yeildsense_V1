# 🌾 YieldSense — Precision Agriculture Yield Forecaster

YieldSense is an end-to-end Machine Learning project designed to predict crop yield using historical agricultural and weather-related data. The main objective of this project is to demonstrate how Machine Learning regression algorithms can be applied to agricultural data to estimate crop yield based on factors such as country or area, crop type, year, rainfall, pesticide usage, and temperature. The project was developed using Python and uses Pandas and NumPy for data processing, Plotly for Exploratory Data Analysis and interactive visualizations, Scikit-learn for Machine Learning preprocessing and regression models, XGBoost for gradient boosting regression, Joblib for saving the trained model, and Streamlit for building the interactive web application. The project follows a complete Machine Learning pipeline starting from dataset collection and data cleaning, followed by Exploratory Data Analysis, feature preparation, categorical feature encoding, train/test splitting, model training, evaluation, model comparison, model selection, model serialization, Streamlit application development, and deployment preparation.

## 🎯 Project Objective

The objective of YieldSense is to build a practical crop-yield prediction system that takes historical agricultural and weather-related information as input and predicts the expected crop yield. The problem is treated as a supervised Machine Learning regression problem because the target variable, crop yield, is a continuous numerical value. Instead of predicting a category such as "high yield" or "low yield", the system predicts an actual numerical yield value. The general Machine Learning workflow can be represented as agricultural and weather data → preprocessing → Machine Learning model → predicted crop yield. The project also demonstrates the importance of comparing different Machine Learning algorithms instead of assuming that a particular algorithm will always perform best.

## 🌱 Real-World Problem

Agricultural production is affected by several factors including weather conditions, rainfall, temperature, pesticide usage, geographical location, crop type, and historical agricultural patterns. Predicting crop yield can help demonstrate how historical data can be used to estimate future agricultural outcomes. YieldSense focuses on building a Machine Learning-based forecasting system using historical crop and environmental information. The system is intended as an educational and portfolio project demonstrating the application of Machine Learning to a real-world agricultural problem.

## 📊 Dataset

The dataset used in YieldSense contains historical agricultural information including Area, Item, Year, crop yield, average annual rainfall, pesticide usage, and average temperature. The dataset contains approximately 28,000 records and includes multiple countries or areas and multiple crop types across different years. The original dataset contains the following important columns: `Area`, `Item`, `Year`, `hg/ha_yield`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, and `avg_temp`. During data preparation, these columns were renamed to simpler names so that they could be used consistently throughout the Machine Learning pipeline. The column `hg/ha_yield` was renamed to `yield`, `average_rain_fall_mm_per_year` was renamed to `rainfall`, `pesticides_tonnes` was renamed to `pesticides`, and `avg_temp` was renamed to `temperature`. The final features used by the Machine Learning models are `Area`, `Item`, `Year`, `rainfall`, `pesticides`, and `temperature`, while `yield` is used as the target variable. The `Area` and `Item` columns are categorical features and therefore require encoding before they can be passed to the Machine Learning models.

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed before model training to understand the dataset and identify relationships between the input features and crop yield. Plotly was used instead of static plotting libraries so that the visualizations remain interactive. The analysis included the distribution of crop yield using a histogram, rainfall versus yield using scatter plots, temperature versus yield using scatter plots, and a numerical correlation heatmap. The EDA stage was used to understand the distribution of the target variable, investigate relationships between environmental variables and crop yield, identify possible patterns, and check the numerical features before training the models. Correlation was treated as an exploratory measure rather than proof of causation because a correlation between two variables does not necessarily mean that one variable causes the other.

## 🧹 Data Preprocessing

The dataset was first loaded using Pandas. Unnecessary index columns were removed when present. The original column names were renamed to make the dataset easier to work with. The target variable was separated from the input features, with `yield` used as the target variable. The categorical features `Area` and `Item` were processed using `OneHotEncoder`. `handle_unknown="ignore"` was used so that the prediction pipeline can safely handle categories that were not present during model training. The preprocessing step was combined with the Machine Learning model using Scikit-learn's `Pipeline` and `ColumnTransformer`. This means that the same preprocessing operations used during training are automatically applied when making predictions in the Streamlit application.

## 🤖 Machine Learning Models

Three regression algorithms were tested in the project: Linear Regression, Random Forest Regressor, and XGBoost Regressor. Linear Regression was used as the baseline model because it provides a simple starting point for a regression problem. Random Forest Regressor was used as an ensemble tree-based model capable of learning nonlinear relationships and interactions between features. XGBoost Regressor was used as a gradient boosting model and provides another powerful approach for tabular regression problems. The purpose of training multiple models was to compare their performance on the same problem rather than assuming beforehand which algorithm would be the best.

## 📏 Evaluation Metrics

The models were evaluated using three main regression metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score. MAE represents the average absolute difference between the actual and predicted values. Lower MAE indicates smaller average prediction errors. RMSE is similar to MAE but gives greater importance to larger prediction errors because the errors are squared before taking the square root. Lower RMSE indicates better performance. R², also called the coefficient of determination, measures how much variation in the target variable is explained by the model. A higher R² generally indicates better predictive performance. The evaluation metrics were calculated using predictions made on data that was not used during model training.

## 🧪 Initial Random Train/Test Evaluation

An initial random train/test split was performed using an 80/20 split. The initial results were approximately Linear Regression with an MAE of 31,791.65, RMSE of 50,757.59, and R² of 0.644824; Random Forest with an MAE of 3,453.36, RMSE of 9,453.84, and R² of 0.987679; and XGBoost with an MAE of 10,462.75, RMSE of 16,650.41, and R² of 0.961780. Although these results showed very strong performance from Random Forest, a random split is not necessarily the most appropriate evaluation method for a forecasting problem because records from different years can be randomly distributed between training and testing data. Therefore, a more realistic time-based evaluation was performed.

## ⏳ Time-Based Validation

Because YieldSense is intended as a forecasting project, a time-based train/test split was used to evaluate how well the models could predict later years using earlier historical years. The data was sorted by year, earlier years were used as the training set, and later years were reserved as the testing set. This approach better represents a real forecasting scenario because the model is trained using past information and evaluated on future information. The time-based evaluation produced the following results: Linear Regression achieved an RMSE of approximately 54,840 and an R² of approximately 0.667, Random Forest achieved an RMSE of approximately 20,275 and an R² of approximately 0.955, and XGBoost achieved an RMSE of approximately 25,236 and an R² of approximately 0.930. Based on this evaluation, Random Forest produced the lowest RMSE and the highest R² among the three tested models. Therefore, Random Forest was selected as the final model for the current version of YieldSense.

## 🌲 Random Forest Model

The Random Forest model was created using multiple decision trees and combined their predictions to produce the final prediction. The initial Random Forest configuration used 200 trees, a fixed random state of 42 for reproducibility, and all available CPU cores for training. Random Forest was selected as the final model based on the time-based evaluation results. Feature importance was also investigated using the trained Random Forest model to understand which processed features contributed most to the model's predictions.

## 🚀 XGBoost Model

XGBoost was included because it is a widely used gradient boosting algorithm for structured and tabular data. The initial XGBoost model used 300 estimators, a learning rate of 0.05, a maximum tree depth of 6, subsampling of 0.8, and column subsampling of 0.8. XGBoost achieved an R² of approximately 0.930 and an RMSE of approximately 25,236 in the time-based evaluation. Although this is a strong result, it was not as strong as the Random Forest result on this particular dataset and evaluation setup. Therefore, XGBoost was not selected as the final deployed model. This comparison demonstrates that a model should be selected based on measured validation performance rather than simply choosing a particular algorithm because it is considered powerful.

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning was also attempted on the Random Forest model. The tuned configuration used 400 trees, a maximum depth of 20, minimum samples split of 2, minimum samples leaf of 1, and `max_features="sqrt"`. However, the tuned model performed worse on the time-based test set than the original Random Forest. The original Random Forest achieved an RMSE of approximately 20,275 and R² of approximately 0.955, while the tuned Random Forest achieved an RMSE of approximately 29,673 and R² of approximately 0.903. Because the tuned model did not improve the validation performance, the original Random Forest model was retained as the final model. This is an important part of the project because hyperparameter tuning does not guarantee better generalization.

## 💾 Model Saving

The final Random Forest pipeline, including the preprocessing steps and trained model, was saved using Joblib. The saved model file is named `yieldsense_random_forest.pkl` and is stored inside the `models` directory. Saving the complete pipeline is important because the Streamlit application needs to apply the same categorical encoding and preprocessing used during model training before making predictions. The saved model allows the application to load the already-trained Machine Learning model without retraining it every time the application starts.

## 🖥️ Streamlit Application

A Streamlit application was developed to provide an interactive interface for YieldSense. The application loads the saved Random Forest model and the dataset, applies the same column transformations used during training, and provides input controls for the user. The user can select a country or area, select a crop, enter a year, enter annual rainfall, enter pesticide usage, and enter average temperature. After the user clicks the prediction button, these values are converted into a Pandas DataFrame with the same feature names expected by the saved Machine Learning pipeline. The model then generates a crop-yield prediction. The application displays the predicted yield, selected crop, selected area, and the input information used for the prediction. The application also displays the prediction in different yield units. The original dataset uses hectograms per hectare (`hg/ha`), and the application converts the prediction into kilograms per hectare and tonnes per hectare for easier interpretation.

## 📈 Streamlit Dashboard

The Streamlit application also provides basic information about the dataset, including the number of records, number of countries or areas, and number of crops. Plotly is used in the project for interactive data analysis and can also be extended in the Streamlit dashboard for additional visualizations. Future dashboard improvements can include actual-versus-predicted yield charts, feature importance charts, historical yield trends, crop comparisons, country comparisons, rainfall versus yield visualizations, and interactive prediction curves.

## 🏗️ System Architecture

The overall YieldSense architecture follows the pipeline: agricultural dataset → data cleaning → exploratory data analysis → feature preparation → categorical encoding → train/test split → Linear Regression, Random Forest, and XGBoost training → evaluation using MAE, RMSE, and R² → time-based validation → final Random Forest model → model serialization using Joblib → Streamlit application → user input → preprocessing → model inference → predicted crop yield. The architecture separates the Machine Learning training process from the application inference process so that the trained model can be reused without retraining whenever the Streamlit application starts.

## 📁 Project Structure

The project follows a simple structure: `YieldSense/` contains the `data/` directory with `yield_df.csv`, the `models/` directory containing `yieldsense_random_forest.pkl`, the `notebooks/` directory containing `01_eda.ipynb`, the `app.py` Streamlit application, `requirements.txt` containing the required Python packages, `README.md` containing the project documentation, and `.gitignore` for excluding unnecessary files from Git. A typical project structure is `YieldSense/data/yield_df.csv`, `YieldSense/models/yieldsense_random_forest.pkl`, `YieldSense/notebooks/01_eda.ipynb`, `YieldSense/app.py`, `YieldSense/requirements.txt`, `YieldSense/README.md`, and `YieldSense/.gitignore`.

## 🛠️ Technologies Used

The project uses Python as the primary programming language. Pandas is used for loading, cleaning, transforming, and analyzing tabular data. NumPy is used for numerical operations and metric calculations. Scikit-learn is used for preprocessing, pipelines, train/test splitting, Linear Regression, Random Forest Regression, and evaluation metrics. XGBoost is used for gradient boosting regression. Plotly is used for interactive Exploratory Data Analysis and model comparison visualizations. Streamlit is used to build the interactive web application. Joblib is used to serialize and load the trained Machine Learning pipeline.

## ⚙️ Installation

To run the project locally, clone the repository and move into the project directory. Create a Python virtual environment using `python3 -m venv venv`, activate it using `source venv/bin/activate` on Linux or macOS, and install the required dependencies using `pip install -r requirements.txt`. After installation, start the Streamlit application using `streamlit run app.py`. The application will normally be available at `http://localhost:8501`.

## 📦 Requirements

The main Python dependencies required for the project are Pandas, NumPy, Scikit-learn, XGBoost, Plotly, Streamlit, and Joblib. These dependencies can be installed using the `requirements.txt` file. For deployment, it is recommended to use a requirements file that matches the environment used during development and model training so that package compatibility issues are minimized.

## 🔮 Future Improvements

YieldSense can be expanded into more advanced versions. V2 can include satellite imagery and satellite-derived features such as NDVI and other vegetation indices to provide additional information about crop conditions. V3 can integrate a real-time weather API so that users can provide or retrieve current weather information instead of manually entering weather values. Additional future improvements can include soil information, time-series features, cross-validation, systematic hyperparameter optimization, regional agricultural analysis, uncertainty estimation, model monitoring, model retraining pipelines, and more advanced forecasting techniques. The project can also be extended to support interactive historical yield trends, crop-specific forecasting, regional comparisons, and automated data pipelines.

## 📌 Important Machine Learning Lessons Demonstrated

YieldSense demonstrates several practical Machine Learning concepts including supervised regression, feature selection, categorical feature encoding, preprocessing pipelines, train/test splitting, time-based validation, model comparison, regression evaluation metrics, nonlinear relationships, ensemble learning, Random Forest, gradient boosting, XGBoost, feature importance, model serialization, interactive visualization, and deployment-oriented Machine Learning development. One of the important lessons from the project is that a model should not be selected only because it is popular or powerful. Linear Regression, Random Forest, and XGBoost were all tested, and the final model was selected based on the actual time-based evaluation results. Another important lesson is that evaluation methodology matters: the initial random split produced extremely high Random Forest performance, but the time-based split produced a more realistic estimate for a forecasting problem. Hyperparameter tuning was also tested, but the tuned Random Forest performed worse, demonstrating that more complex parameters do not automatically produce better generalization.

## ⚠️ Limitations

The current version of YieldSense is based on historical country-level agricultural data and a limited number of environmental and agricultural features. It does not currently use detailed field-level soil measurements, satellite imagery, real-time weather observations, or high-resolution geographic information. The model should therefore not be interpreted as a professional agricultural decision-making system. Model performance depends heavily on the quality, coverage, and distribution of the training data. Predictions for areas, crops, years, or environmental conditions that are substantially different from the training data may be less reliable.

## ⚠️ Disclaimer

YieldSense is an educational and portfolio Machine Learning project created to demonstrate an end-to-end regression workflow for agricultural data. Its predictions are estimates generated by a Machine Learning model and should not be treated as guaranteed agricultural outcomes, professional farming advice, financial advice, or a replacement for expert agricultural analysis.

## 👨‍💻 Author

**Sushil Singh** — B.Tech Artificial Intelligence & Machine Learning

YieldSense demonstrates an end-to-end Machine Learning workflow from raw agricultural data and Exploratory Data Analysis to model comparison, time-based validation, model serialization, interactive Streamlit prediction, and deployment preparation.
