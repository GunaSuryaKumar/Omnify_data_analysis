# Omnify_data_analysis
data analysis task
# 📊 Booking Revenue Dashboard – Omnify Data Analysis

This project is an **interactive dashboard** built using **Dash**, **Plotly**, and **Flask** to analyze booking and revenue trends for a multi-service business. It allows users to explore booking patterns, analyze service-based revenues, and forecast future revenue using regression models.

---

## 🚀 Features

- 📈 **Revenue Trends** by Month and Service Type
- 🧩 **Booking Type Distribution** (Classes, Parties, Rentals, Subscriptions)
- ⏰ **Peak Booking Hours** visualization
- 🔍 **Interactive Filtering** by Booking Type
- 📉 **Revenue Forecasting** using Linear and Polynomial Regression

---

## 📁 Project Structure

BookingDashboard/ ├── app.py # Main Flask + Dash application ├── DataAnalyst_Assesment_Dataset.xlsx # Input dataset ├── templates/ │ └── index.html # Landing page (Flask) ├── static/ │ └── style.css # Styling for landing page ├── requirements.txt # Python dependencies └── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/BookingDashboard.git
cd BookingDashboard
2. Set up a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Add the dataset
Place the file DataAnalyst_Assesment_Dataset.xlsx in the root directory. (This file is not included in the repo due to size/privacy.)

5. Run the application
bash
Copy
Edit
python app.py
Open your browser and navigate to: http://localhost:8050

📊 Dashboard Tabs Overview
Revenue Overview: Monthly revenue trends.

Service Revenue: Revenue per service category.

Booking Analysis: Distribution by type and booking hours.

Interactive Booking Trend: Filterable revenue trends by booking type.

Forecast Revenue: Predict future revenue using selected models and timeframes.

🔮 Forecasting Models
You can choose from:

Linear Regression

Polynomial Regression (Degree 2 & 3)

Adjust the forecast horizon (future days to predict) and the training window (how many past days to use for learning).

🛠️ Dependencies
All dependencies are listed in requirements.txt. Key libraries include:

Dash (dash, dash-core-components, dash-html-components)

Plotly

Flask

Pandas

NumPy

Scikit-learn

🧠 Insights
This dashboard was developed as part of a data analysis task for Omnify. It provides:

Clear visual insights for stakeholders

Actionable analytics for booking strategy

Predictive trends to aid planning

