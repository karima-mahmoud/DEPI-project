# Australian Vehicle Price Prediction 🚗💰

This repository contains a Streamlit application for predicting vehicle prices based on a wide range of features. The project utilizes machine learning techniques to analyze and predict the price of vehicles in the Australian market.

## 🌟 Key Features

* **Interactive Price Prediction**: Enter vehicle details to get an estimated market price.
* **Comprehensive Data Cleaning**: Automated data preprocessing for accurate predictions.
* **Visual Analytics**: Explore vehicle data with interactive visualizations.
* **Machine Learning Models**: Compare the performance of various regression algorithms.

## 📁 Project Structure

```
.
├── Model.py          # Machine learning model code
├── Visualizations.py # Data visualization functions
├── app.py            # Main Streamlit application file
├── requirements.txt  # Required Python packages
├── README.md         # Project documentation (this file)
└── Australian Vehicle Prices.csv # Sample dataset
```

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

* Python 3.9+
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* gdown (for Google Drive file downloads)

Install the required packages using:

```bash
pip install -r requirements.txt
```

### 💾 Dataset

The application uses the **Australian Vehicle Prices** dataset, which contains detailed vehicle listings with features like brand, year, engine size, fuel type, kilometers driven, and price. Make sure the dataset is placed in the project directory as `Australian Vehicle Prices.csv` or update the file path accordingly.

### 🏁 Running the Application

Run the Streamlit app with the following command:

```bash
streamlit run app.py
```

The app should now be accessible at **[http://localhost:8501](http://localhost:8501)** in your browser.

## 📊 Model Training

The project supports multiple regression models, including:

* Linear Regression
* Ridge
* Lasso
* Random Forest
* Gradient Boosting
* Extra Trees

The `Model.py` file handles model training and prediction, while the `Visualizations.py` file contains the data visualization logic.

## 🎨 Key App Pages

* **Home Page**: Overview of the project and dataset.
* **Visualizations**: Interactive data exploration.
* **Model**: Vehicle price prediction based on user inputs.

## 📌 To-Do

* [ ] Add model fine-tuning
* [ ] Implement more robust error handling
* [ ] Optimize data preprocessing
* [ ] Enhance UI/UX with modern design elements


## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

## 🌐 Resources

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
* [Plotly Documentation](https://plotly.com/python/)

## ❤️ Support

If you find this project helpful, please ⭐ the repository!

---

## 🙋‍♂️ Author

**👨‍💻 Karima Mahmoud**  
📫 karimamahmoudsalem1@gmail.com  
🐙 GitHub: https://github.com/karima-mahmoud
