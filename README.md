# 🩺 MedAdvisorAI: Multi-Agent Heart Disease Prediction and Personalized Healthcare Recommendation System

MedAdvisorAI is a machine learning-based healthcare application designed to support preliminary cardiovascular risk assessment and personalized healthcare guidance.

The system combines a **HistGradientBoosting Classifier**, a multi-agent recommendation framework, a central Recommendation Coordinator, and a Django-based web application to provide a unified approach to cardiovascular risk prediction and healthcare recommendations.

---

## 📌 Project Overview

Cardiovascular diseases are a major global health concern. Early identification of potential risk factors and improved understanding of health information can help individuals make more informed decisions about their well-being.

MedAdvisorAI explores the integration of Artificial Intelligence, Machine Learning, and Explainable Artificial Intelligence to support preliminary cardiovascular risk assessment.

### Key Highlights

- Analyzes patient health parameters for cardiovascular risk prediction.
- Uses a HistGradientBoosting Classifier for machine learning-based classification.
- Organizes recommendations through five specialized agents.
- Coordinates recommendation outputs through a central Recommendation Coordinator.
- Integrates the prediction and recommendation workflow into a Django web application.
- Uses SQLite for database management.
- Supports model evaluation using standard classification metrics.

**Project Domain:** Artificial Intelligence, Machine Learning, Healthcare Technology, and Web Development.

---

## 🎯 Problem Statement

Traditional approaches to preliminary cardiovascular risk assessment may require users to interpret multiple health indicators and obtain healthcare guidance from different sources.

- Patients may find it difficult to understand the significance of individual health parameters.
- General healthcare recommendations may not address the specific risk factors identified in a patient's health information.
- Machine learning predictions alone may not provide practical guidance for users.
- Healthcare recommendations related to diet, exercise, medication adherence, and lifestyle may be distributed across different sources.
- A unified system can help present preliminary cardiovascular risk information alongside organized healthcare recommendations.

**MedAdvisorAI aims to address these challenges by integrating machine learning-based cardiovascular risk prediction with a multi-agent personalized healthcare recommendation framework.**

---

## 🎯 Project Objectives

- Develop a machine learning model for preliminary cardiovascular risk prediction.
- Collect and process relevant patient health information.
- Perform data preprocessing and feature engineering.
- Integrate the HistGradientBoosting Classifier into the application.
- Develop specialized agents for healthcare recommendation categories.
- Implement a central Recommendation Coordinator to organize agent outputs.
- Integrate patient profile management, prediction, and recommendations into a Django web application.
- Present health information and assessment results through a dashboard.
- Evaluate model performance using appropriate classification metrics.
- Explore Explainable AI and other enhancements to improve the system in future development.

---

## ✨ Key Features

### 🧠 Cardiovascular Risk Prediction

- Uses a HistGradientBoosting Classifier for cardiovascular risk prediction.
- Processes patient health parameters to generate a model prediction.
- Supports preliminary assessment based on the supplied health information.
- Provides a foundation for evaluating cardiovascular risk classification.

### 👤 Patient Profile Management

- Organizes relevant patient health information.
- Supports the collection and processing of patient health parameters.
- Provides a structured foundation for the prediction and recommendation workflow.

### 🤖 Multi-Agent Recommendation Framework

The application incorporates five specialized recommendation agents.

- **Diet Recommendation Agent:** Provides diet-related guidance.
- **Exercise Recommendation Agent:** Organizes physical activity recommendations.
- **Medication Recommendation Agent:** Handles medication-related guidance, including adherence information where implemented.
- **Lifestyle Recommendation Agent:** Organizes recommendations related to lifestyle modifications.
- **Cardiovascular Recommendation Agent:** Focuses on general cardiovascular health guidance.

### 🎛️ Recommendation Coordination

- Uses a central Recommendation Coordinator to organize outputs from the specialized agents.
- Combines recommendation categories into a unified response.
- Supports coordinated and personalized healthcare guidance.
- Provides a structured approach to presenting recommendations.

### 📊 Dashboard and Results

- Presents patient health information supported by the application.
- Displays cardiovascular prediction results.
- Organizes the available recommendations.
- Provides a unified view of assessment information and recommendation outputs.

### 🗄️ Database Management

- Uses SQLite as the database technology.
- Supports database operations required by the application.
- Integrates with the Django application where configured.

### 🔍 Explainable AI — Planned Enhancement

- Explore techniques such as SHAP and LIME to explain model predictions.
- Investigate the contribution of individual features to cardiovascular risk predictions.
- Improve the interpretability of model outputs.

*Explainable AI functionality should be described as implemented only if the corresponding code is present and working.*

---

## 🔬 Methodology

The methodology follows the headings used in the MedAdvisorAI project report.

### 👤 Patient Data Collection and Profile

- Collect patient health parameters required by the system.
- Organize available demographic, clinical, and lifestyle information.
- Create a structured patient health profile.
- Prepare the profile for data preprocessing and prediction.

### 🧹 Data Preprocessing

- Validate patient inputs.
- Handle missing values according to the implemented preprocessing strategy.
- Process numerical and categorical data.
- Prepare health information for feature engineering and model prediction.

### ⚙️ Feature Engineering

- Identify and prepare the features required by the prediction model.
- Generate relevant cardiovascular features where supported by the implementation.
- Prepare numerical and categorical features in the expected format.
- Maintain consistency between the training data and prediction inputs.

### ❤️ Cardiovascular Risk Prediction

- Use the HistGradientBoosting Classifier for cardiovascular risk prediction.
- Process the prepared patient health features.
- Generate predictions using the trained model.
- Calculate or display cardiovascular risk probability if supported by the implementation.
- Classify risk levels according to the implemented prediction logic.

### 🤖 Multi-Agent Recommendation Framework

- Develop specialized agents for different healthcare recommendation categories.
- Generate diet-related guidance through the Diet Recommendation Agent.
- Organize exercise-related guidance through the Exercise Recommendation Agent.
- Handle medication-related guidance through the Medication Recommendation Agent.
- Generate lifestyle guidance through the Lifestyle Recommendation Agent.
- Organize cardiovascular health recommendations through the Cardiovascular Recommendation Agent.

### 🎛️ Recommendation Coordination

- Implement a central Recommendation Coordinator.
- Collect outputs from the specialized recommendation agents.
- Organize the recommendations into a unified structure.
- Reduce redundant recommendations where supported by the implementation.
- Prepare the coordinated recommendations for presentation.

### 🌐 Web Application Integration

- Integrate patient profile management into the Django application.
- Connect the health assessment workflow to the prediction model.
- Integrate the multi-agent recommendation framework.
- Connect the Recommendation Coordinator to the application workflow.
- Present prediction results and recommendations through the web interface.

### 📊 Dashboard and Results

- Display patient health information supported by the application.
- Present relevant health indicators where implemented.
- Display cardiovascular prediction results.
- Show risk probability and risk categories if supported by the application.
- Organize the recommendations generated by the agents.

### 🔍 Explainable AI and Reports

- Explore Explainable AI techniques such as SHAP and LIME.
- Investigate feature contributions to model predictions.
- Explore PDF report generation to present structured assessment results.

*SHAP/LIME integration and PDF report generation should be identified as planned functionality if they are not yet implemented.*

### 🧪 Testing and Evaluation

- Test the application with valid patient profiles.
- Verify how the application handles missing or invalid inputs.
- Test the machine learning prediction workflow.
- Verify the output of the recommendation agents.
- Test the Recommendation Coordinator.
- Evaluate the model using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Assess recommendation relevance and personalization using suitable evaluation methods where available.

### 🚀 Deployment and Future Enhancement

- Prepare the application for deployment.
- Improve model performance and generalization.
- Explore more diverse and clinically validated datasets.
- Investigate advanced recommendation agents and Explainable AI techniques.
- Consider real-time health monitoring and additional healthcare integrations.
- Explore further improvements to accessibility and personalization.

### 🔄Proposed System — Architecture & Flow
<img width="644" height="823" alt="image" src="https://github.com/user-attachments/assets/644c8f0f-10d2-4c3d-be88-f0955075da80" />

## 🏗️ System Architecture

MedAdvisorAI integrates patient data processing, machine learning-based cardiovascular risk prediction, a multi-agent recommendation framework, and a Django web application.

<img width="991" height="661" alt="image" src="https://github.com/user-attachments/assets/99b632fa-77c8-4dcb-b294-f5dab894782b" />

### 🧩 Architecture Components

- **Frontend Layer**
  - HTML, CSS, and JavaScript.
  - Collects supported user inputs and displays results.

- **Backend Layer**
  - Python and Django.
  - Handles application requests and connects the system components.

- **Machine Learning Layer**
  - Scikit-learn.
  - Uses the HistGradientBoosting Classifier for prediction.

- **Recommendation Layer**
  - Python-based specialized recommendation agents.
  - Organizes recommendations across the five healthcare categories.

- **Coordination Layer**
  - Central Recommendation Coordinator.
  - Organizes outputs from the specialized agents.

- **Database Layer**
  - SQLite.
  - Supports application data storage.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Django |
| Machine Learning | Scikit-learn |
| Classification Model | HistGradientBoosting Classifier |
| Database | SQLite |
| Recommendation Framework | Python-based specialized agents |
| Recommendation Coordination | Central Recommendation Coordinator |
| Model Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC |

---

## 📂 Project Structure

The following is a representative Django project structure. Replace the placeholder names with the actual directories and filenames in your repository.

```text
MedAdvisorAI/
│
├── manage.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── <django_project>/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── <application>/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── templates/
│
├── <machine_learning>/
│   ├── model_training.py
│   ├── prediction.py
│   └── <saved_model_file>
│
├── <recommendation_agents>/
│   ├── diet_agent.py
│   ├── exercise_agent.py
│   ├── medication_agent.py
│   ├── lifestyle_agent.py
│   ├── cardiovascular_agent.py
│   └── recommendation_coordinator.py
│
├── db.sqlite3
│
└── <static_files>/
    ├── css/
    ├── js/
    └── images/
```

*This structure is illustrative, not a verified listing of your repository. Update it to match your actual files. Do not commit a database containing private patient information.*

---

## 🚀 Installation and Setup

### 📋 Prerequisites

- Python installed on your system.
- Git installed on your system.
- A code editor such as Visual Studio Code.
- A terminal or command prompt.
- The project dependencies.

### 📥 Clone the Repository

```bash
git clone https://github.com/DevTech-SR09/MedAdvisorAI.git
```

Navigate to the project directory:

```bash
cd MedAdvisorAI
```

### 🐍 Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 📦 Install Dependencies

If `requirements.txt` is available, run:

```bash
pip install -r requirements.txt
```

### 🗄️ Configure the Database

The project uses SQLite according to the specified technology stack.

If the application uses Django migrations, run:

```bash
python manage.py migrate
```

### ⚙️ Check the Configuration

- Confirm that Django settings are configured correctly.
- Verify that the trained model file is available if the application loads a saved model.
- Check that all required dependencies are installed.
- Confirm that the database configuration matches the project.

---

## ▶️ Running the Application

### 🖥️ Start the Django Development Server

From the directory containing `manage.py`, run:

```bash
python manage.py runserver
```

Open the following address in your browser:

```text
http://127.0.0.1:8000/
```

### 🩺 Typical Application Workflow

- Open the application in your browser.
- Enter the patient health information requested by the interface.
- Submit the information for assessment.
- Review the prediction returned by the machine learning model.
- Review the available recommendations generated by the recommendation framework.
- Use the results as preliminary information, not as a confirmed medical diagnosis.

*The actual workflow depends on the pages and features implemented in your application.*

---

## 🧪 Testing and Evaluation

### 🧠 Model Evaluation Metrics

The project report identifies the following classification metrics for evaluating the model.

- **Accuracy:** Measures the proportion of correctly classified instances.
- **Precision:** Measures how many positive predictions are correct.
- **Recall:** Measures how many actual positive instances are identified.
- **F1-score:** Combines precision and recall into a single measure.
- **ROC-AUC:** Measures the model's ability to distinguish between classes, where applicable.

### 🤖 Recommendation Framework Testing

- Test the outputs of the specialized recommendation agents.
- Verify that the Recommendation Coordinator organizes the outputs correctly.
- Check the consistency and relevance of recommendations.
- Test how the system handles missing or unexpected input values.

### 🌐 Web Application Testing

- Verify that the Django application starts successfully.
- Test patient input forms and validation.
- Confirm that the prediction workflow operates correctly.
- Verify that prediction results and recommendations are displayed as expected.
- Test error handling for invalid inputs and missing model files.

### 📊 Evaluation Results

Document the actual evaluation results when they are available.

- Dataset and number of records.
- Training and testing methodology.
- Accuracy, Precision, Recall, and F1-score.
- ROC-AUC, where applicable.
- Observed limitations.
- Evaluation of recommendation relevance and personalization, if performed.

**Do not report model scores unless they have been measured using your actual implementation.**

---

## 🔮 Future Enhancements

The following future enhancements follow the topics in **Chapter 8.2 of the MedAdvisorAI project report**.

### 🎯 Improved Prediction Accuracy

- Explore advanced machine learning and deep learning algorithms to improve cardiovascular risk prediction.
- Improve the model's ability to generalize to new patient data.
- Evaluate alternative models using appropriate validation methods.

### 📊 Larger and Diverse Datasets

- Incorporate more diverse, clinically validated datasets.
- Improve model robustness across different patient populations.
- Investigate the performance of the model on additional patient data.

### 🤖 Advanced Multi-Agent Framework

- Develop more sophisticated recommendation agents.
- Improve the generation of context-aware healthcare guidance.
- Strengthen coordination between the specialized recommendation agents.

### 🔍 Enhanced Explainable AI

- Integrate advanced Explainable AI techniques such as SHAP and LIME.
- Improve the interpretability of model predictions.
- Identify and present the features contributing to cardiovascular risk predictions.

### ⌚ Real-Time Health Monitoring

- Explore integration with wearable devices and health sensors.
- Collect health measurements such as heart rate and physical activity.
- Investigate the use of available real-time health information in the assessment workflow.

### 📱 Mobile Application Development

- Develop an Android or iOS application.
- Make cardiovascular risk assessment more accessible.
- Provide access to personalized recommendations through a mobile interface.

### 🏥 Electronic Health Record Integration

- Explore integration with electronic health records.
- Support analysis of relevant patient medical history with appropriate authorization.
- Apply suitable privacy and security safeguards when handling health information.

### 🎯 Personalized Recommendation Optimization

- Improve recommendation generation using individual health conditions.
- Consider lifestyle habits and user preferences.
- Explore the use of changing health parameters to improve recommendation relevance.

---

## 📚 Key Learning Outcomes

- **Machine Learning:** Working with a classification model for cardiovascular risk prediction.
- **Data Preprocessing:** Preparing patient health information for model inference.
- **Feature Engineering:** Organizing relevant features for prediction.
- **Django Development:** Integrating backend functionality into a web application.
- **Multi-Agent Systems:** Separating healthcare recommendation responsibilities into specialized agents.
- **Recommendation Coordination:** Organizing outputs from different recommendation components.
- **Database Management:** Using SQLite with a Django-based application.
- **Model Evaluation:** Understanding Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- **Responsible AI:** Recognizing the importance of interpretability, privacy, and professional oversight in healthcare applications.

---

## 🏁 Conclusion

MedAdvisorAI combines machine learning-based cardiovascular risk prediction with a multi-agent healthcare recommendation framework.

The system brings together patient health information processing, feature engineering, a HistGradientBoosting Classifier, specialized recommendation agents, a central Recommendation Coordinator, and a Django-based web application.

The project provides a foundation for further exploration of personalized healthcare guidance and AI-assisted preliminary cardiovascular risk assessment.

Future development can focus on improving prediction accuracy, incorporating larger and more diverse datasets, enhancing the multi-agent framework, integrating Explainable AI, supporting real-time health monitoring, developing a mobile application, integrating electronic health records, and optimizing personalized recommendations.

MedAdvisorAI is intended to support health awareness and complement professional medical services. It is not a substitute for clinical diagnosis, treatment, or consultation with qualified healthcare professionals.

---

## ⚠️ Medical Disclaimer

- This project is intended for educational and research purposes.
- Predictions generated by the system are not confirmed medical diagnoses.
- The model may produce inaccurate or incomplete results.
- Recommendations should not replace advice from qualified healthcare professionals.
- Do not start, stop, or change medication based on the application's output.
- Consult a qualified healthcare professional for diagnosis, treatment, and medical concerns.

---

## 👨‍💻 Author

**GitHub Profile:** [DevTech-SR09](https://github.com/DevTech-SR09)

**Project Repository:** [MedAdvisorAI](https://github.com/DevTech-SR09/MedAdvisorAI)

---

⭐ If you find this project interesting, explore the repository and follow its development.
