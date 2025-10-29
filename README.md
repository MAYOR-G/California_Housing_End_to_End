# California Housing Price Prediction: An End-to-End ML Project

This project implements a comprehensive machine learning pipeline for predicting California housing prices, from data acquisition and model training to deployment as a robust FastAPI web service. It serves as an excellent example of an end-to-end MLOps workflow.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Model](#training-the-model)
  - [Running the FastAPI Application](#running-the-fastapi-application)
  - [Accessing the Web Interface](#accessing-the-web-interface)
  - [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

The primary objective of this project is to accurately predict the median house value in various California districts. Predictions are based on key features such as median income, house age, average number of rooms, population density, and geographical coordinates. The project encompasses the following stages:

1.  **Data Acquisition**: Leverages the `fetch_california_housing` dataset from `scikit-learn` for reliable data sourcing.
2.  **Exploratory Data Analysis (EDA)**: Conducted within a Jupyter Notebook to visualize target distributions, analyze feature correlations, and gain insights into the dataset.
3.  **Model Training**: Utilizes a `HistGradientBoostingRegressor` for its efficiency and strong predictive performance on tabular data.
4.  **Model Persistence**: The trained model is serialized using `joblib` for efficient storage and retrieval.
5.  **Web Service Deployment**: The model is exposed via a high-performance FastAPI application, offering both a user-friendly web interface and a programmatic RESTful API.

## Features

-   **Automated Data Loading**: Seamless integration with `scikit-learn` for fetching the California housing dataset.
-   **Interactive EDA & Model Training**: Dedicated Jupyter notebook (`california_train.ipynb`) for in-depth data exploration, visualization, and model development.
-   **Robust Machine Learning Model**: Employs `HistGradientBoostingRegressor` for accurate and efficient price predictions.
-   **Model Serialization**: Ensures easy saving and loading of the trained model using `joblib`.
-   **High-Performance FastAPI Web Service**:
    -   Intuitive web interface for real-time predictions.
    -   Comprehensive RESTful API with clear documentation (via Swagger UI/ReDoc, provided by FastAPI).
    -   Dedicated health check endpoint for monitoring.
-   **Containerization with Docker**: Facilitates consistent and portable deployment across various environments.
-   **CI/CD Integration**: Includes GitHub Actions workflow (`cicd.yml`) for automated testing and deployment (if fully implemented).

## Tech Stack

This project is built using a modern and efficient tech stack:

| Category            | Technology to Python for data science.
| Python | Machine Learning | [![Python](https://img.shields.io/badge/Python-3776AB?style=for-badge&logo=python&logoColor=white)](https://www.python.org/) |
| scikit-learn | Machine Learning Library | [![scikit-learn](https://img.io/badge/scikit--learn-F7931E?style=for-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) |
| FastAPI | Web Framework | [![FastAPI](https://img.io/badge/FastAPI-009688?style=for-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) |
| Uvicorn | ASGI Server | [![Uvicorn](https://img.io/badge/Uvicorn-593585?style=for-badge&logo=uvicorn&logoColor=white)](https://www.uvicorn.org/) |
| Jupyter Notebook | Interactive Computing | [![Jupyter](https://img.io/badge/Jupyter-F37626?style=for-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) |
| Docker | Containerization | [![Docker](https://img.io/badge/Docker-2496ED?style=for-badge&logo=docker&logoColor=white)](https://www.docker.com/) |
| GitHub Actions | CI/CD | [![GitHub Actions](https://img.io/badge/GitHub%20Actions-2088FF?style=for-badge&logo=github-actions&logoColor=white)](https://docs.github.com/en/actions) |
| uv | Dependency Management | [![uv](https://img.io/badge/uv-6C5CE7?style=for-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJDNi40NzcgMiAyIDYuNDc3IDIgMTJDMiAxNy41MjMgNi40NzcgMjIgMTIgMjJDMTcuNTIzIDIyIDIyIDE3LjUyMyAyMiAxMkMyMiA2LjQ3NyAxNy41MjMgMiAxMiAyWk0xMiA0QzE2LjQxOCA0IDIwIDcuNTgyIDIwIDEyQzIwIDE2LjQxOCAxNi40MTggMjAgMTIgMjBDNy41ODIgMjAgNCAxNi40MTggNCAxMkM0IDcuNTgyIDcuNTgyIDQgMTIgNFpNMTIgN0M5LjIyNCA3IDcgOS4yMjQgNyAxMkM3IDE0Ljc3NiA5LjIyNCAxNyAxMiAxN0MxNC43NzYgMTcgMTcgMTQuNzc2IDE3IDEyQxcgOS4yMjQgMTQuNzc2IDcgMTIgN1pNMTIgOUMxMy42NTcgOSAxNSAxMC4zNDMgMTUgMTJDMTUgMTMuNjU3IDEzLjY1NyAxNSAxMiAxNUMxMC4zNDMgMTUgOSAxMy42NTcgOSAxMkM5IDEwLjM0MyAxMC4zNDMgOSAxMiA5WiIgZmlsbD0iY3VycmVudENvbG9yIi8+Cjwvc3ZnPgo=)](https://github.com/astral-sh/uv) |

## Project Structure

```
California_Housing_End_to_End/
├── .github/
│   └── workflows/
│       └── cicd.yml             # GitHub Actions workflow for CI/CD
├── templates/
│   └── index.html               # Frontend HTML for the web interface
├── .gitignore                   # Specifies intentionally untracked files to ignore
├── .python-version              # Defines the Python version (e.g., for pyenv)
├── california_train.ipynb       # Jupyter notebook for model training and EDA
├── Dockerfile                   # Dockerfile for containerizing the application
├── main.py                      # FastAPI application entry point
├── model.joblib                 # Trained machine learning model (generated after training)
├── pyproject.toml               # Project metadata and dependencies (Poetry/Rye/uv)
├── README.md                    # Project README file
└── uv.lock                      # Dependency lock file (generated by uv)
```

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

Ensure you have the following installed:

-   **Python 3.8+**: The core language for the project.
-   **`uv`**: A fast Python package installer and resolver (recommended). Alternatively, `pip` and `venv` can be used.
-   **Docker**: (Optional) Required for building and running the containerized application.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/California_Housing_End_to_End.git
    cd California_Housing_End_to_End
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```
    If `uv` is not installed or preferred, use `pip` with a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt # You might need to generate requirements.txt from pyproject.toml first
    ```
    *(Note: `pyproject.toml` is present, indicating dependency management via tools like Poetry, Rye, or uv. `uv sync` is the recommended approach here. If `requirements.txt` is missing, generate it from `pyproject.toml` using your preferred tool, e.g., `rye export -o requirements.txt` or `poetry export -f requirements.txt --output requirements.txt --without-hashes`)*

### Training the Model

The machine learning model is developed and trained within a Jupyter notebook.

1.  **Start Jupyter Lab/Notebook:**
    ```bash
    uv run jupyter lab
    ```
    or, if using a virtual environment:
    ```bash
    source .venv/bin/activate # if using venv
    jupyter lab
    ```

2.  **Execute `california_train.ipynb`:**
    Open the `california_train.ipynb` notebook and run all cells sequentially. This process will:
    -   Load the California housing dataset.
    -   Perform comprehensive Exploratory Data Analysis (EDA).
    -   Train a `HistGradientBoostingRegressor` model.
    -   Evaluate the model's performance metrics.
    -   Persist the trained model as `model.joblib` in the project root directory.

### Running the FastAPI Application

Once the model is trained and `model.joblib` is available in the project root, you can launch the FastAPI application.

1.  **Run the application using `uvicorn`:**
    ```bash
    uv run uvicorn main:app --reload
    ```
    or, if using a virtual environment:
    ```bash
    source .venv/bin/activate # if using venv
    uvicorn main:app --reload
    ```
    The application will become accessible at `http://127.0.0.1:8000`.

### Accessing the Web Interface

Navigate to the following URL in your web browser:
```
http://127.0.0.1:8000/
```
This will render the interactive web interface (`templates/index.html`), allowing you to submit housing features and receive predictions.

### API Endpoints

The FastAPI application provides the following RESTful API endpoints:

-   **`GET /health`**:
    -   **Description**: Health check endpoint to verify API operational status.
    -   **Response**: `{"status": "ok"}`

-   **`GET /`**:
    -   **Description**: Serves the main web interface for user interaction.
    -   **Response**: Renders `templates/index.html`.

-   **`POST /api/predict`**:
    -   **Description**: Predicts the median house value based on a set of input features.
    -   **Request Body (JSON Example)**:
        ```json
        {
          "MedInc": 3.87,       // Median income in 10k USD
          "HouseAge": 29.0,     // Median house age
          "AveRooms": 6.2,      // Average number of rooms
          "AveBedrms": 1.0,     // Average number of bedrooms
          "Population": 1200,   // Block population
          "AveOccup": 2.8,      // Average household occupancy
          "Latitude": 34.0,     // House block latitude
          "Longitude": -118.0   // House block longitude
        }
        ```
    -   **Successful Response (JSON Example)**:
        ```json
        {
          "prediction_usd": 250000.00, // Predicted median house value in USD
          "inputs": { ... }           // Echoes the input data for verification
        }
        ```
    -   **Error Responses**:
        -   `422 Unprocessable Entity`: Indicates input validation failures (e.g., incorrect data types, missing fields).
        -   `500 Internal Server Error`: Catches unexpected server-side issues, such as a missing model file.

## Model Details

The core predictive component is a `HistGradientBoostingRegressor` from the `scikit-learn` library. This model is configured with `learning_rate=0.08` and `max_depth=None`, offering a robust solution for tabular regression tasks. It inherently handles missing values and can effectively manage various feature types without extensive preprocessing.

## Deployment

The project is designed for easy deployment using Docker. The provided `Dockerfile` containerizes the FastAPI application, ensuring a consistent runtime environment across different platforms (e.g., local machine, cloud VMs, Kubernetes).

To build the Docker image:
```bash
docker build -t california-housing-predictor .
```

To run the Docker container:
```bash
docker run -p 8000:8000 california-housing-predictor
```
**Important**: Ensure that the `model.joblib` file is present in the project root directory *before* building the Docker image, as it is copied into the container during the build process.

## Contributing

We welcome contributions to enhance this project! Please feel free to:
-   **Open an issue**: To report bugs, suggest features, or ask questions.
-   **Submit a pull request**: To propose code changes, improvements, or new functionalities.

## License

This project is open-sourced under the MIT License. For full details, please refer to the `LICENSE` file (if available in the repository).

## Contact

For any inquiries, feedback, or support, please open an issue on the project's GitHub repository.
