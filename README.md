# California Housing Linear Regression API

## Overview

This project implements **Linear Regression from scratch** using NumPy and deploys it as a REST API using FastAPI.
The model is trained on the California Housing dataset and can generate predictions based on user input features.

---

## Features

* Implemented Linear Regression **without using sklearn**
* Manual feature scaling using mean and standard deviation
* Model serialization using Pickle (`.pkl`)
* REST API built with FastAPI
* Docker support for containerized deployment

---

## Tech Stack

* Python
* NumPy
* FastAPI
* Uvicorn
* Docker

---

## Project Structure

```
linearregressionAPI/
│
├── app.py              # FastAPI application
├── linear_model.pkl    # Saved model parameters
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .dockerignore
```

---

## How It Works

1. Input features are received via API
2. Features are standardized using training mean and std
3. Bias term is added
4. Prediction is computed using dot product with learned parameters

---

## Running Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Start the server

```
uvicorn app:app --reload
```

### 3. Open in browser

```
http://127.0.0.1:8000/docs
```

Use the interactive UI to test predictions.

---

## API Endpoint

### POST `/predict`

**Input:**

```
{
  "features": [8 numerical values]
}
```

**Output:**

```
{
  "prediction": value
}
```

---

## Future Improvements

* Compare with sklearn implementation
* Improve input validation
* Deploy API online (Render / AWS)

