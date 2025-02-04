# Video Game Sales Analyzer
A Flask web application to analyze video game sales data which includes historical sales analysis, regional distribution of games, and machine learning-based sales predictions.

## About The Project
This Video Game Sales Analyzer was developed as part of my journey to understand data analysis and web development. 
The project combines my passion for games with practical data science skills, creating a simple but effective interactive platform to explore historical video game sales trends.

## Dataset
* **Source**: [Video Games Sales Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales) from Kaggle
* **Creator**: Gregory Smith
* **Contents**: 16,500+ video games with sales data
  
## Features:
- Historical sales analysis
- Regional sales distribution
- Sales prediction using Linear Regression

## Tech Stack:
- Flask (Web Framework)
- SQLite (Database)
- Pandas (Data Processing)
- Matplotlib (Visualization)
- Scikit-learn (Machine Learning)

## Prerequisites: 
- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation Guide: 
1. Clone the Repository
```bash
git clone https://github.com/yourusername/video-game-sales-analyzer.git
cd video-game-sales-analyzer
```

2. Virtual Environment Setup
Create Virtual Environment
```bash
python -m venv venv
```

Activate Virtual Environment
For Windows:
```bash
venv\Scripts\activate
```

For macOS/Linux:
```bash
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Run the Application
```bash
flask run
```

The application will be available at http://localhost:5000
