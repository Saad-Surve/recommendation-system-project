# Article Recommendation System

A personalized article recommendation system that suggests articles based on user preferences and reading history.

## Project Structure

```
├── api.py                  # Backend API server
├── frontend/              # Frontend application
│   ├── app.py            # Frontend FastAPI server
│   ├── static/           # Static assets (CSS, JS, images)
│   └── templates/        # HTML templates
├── db_manager.py          # Database management utilities
├── db_recommender.py      # Recommendation engine
├── recommender.py         # Core recommendation algorithms
├── requirements.txt       # Project dependencies
└── data/                 # Dataset directory
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository and navigate to the project directory

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

The application requires two servers to be running simultaneously:

1. Start the Backend API server (in the root directory):
```bash
python api.py
```
The API server will start on `http://localhost:5000`

2. Start the Frontend server (in a new terminal):
```bash
cd frontend
python app.py
```
The frontend will be available at `http://localhost:8001`

## Features

- User authentication and registration
- Personalized article recommendations
- User preference management
- Article interaction tracking
- Interactive web interface

## Database

The system uses SQLite database (`recommender.db`) which:
- Stores user information and preferences
- Maintains article data
- Tracks user interactions
- Handles recommendation history

## Troubleshooting

1. If a port is already in use:
   - Close any running Python processes
   - Check if another application is using the port
   - Try restarting your terminal

2. Database issues:
   - Make sure `recommender.db` exists in the root directory
   - Check file permissions
   - Try deleting and letting the system recreate the database

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request