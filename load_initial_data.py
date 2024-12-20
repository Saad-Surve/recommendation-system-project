import pandas as pd
from db_manager import DBManager
from datetime import datetime
import sqlite3
import json
import time
from passlib.context import CryptContext

# Password hashing setup
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

def get_db(max_retries=5, retry_delay=1):
    """Get database connection with retry logic"""
    for attempt in range(max_retries):
        try:
            return sqlite3.connect('recommender.db', timeout=20.0)
        except sqlite3.OperationalError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Database locked, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

def load_initial_data():
    try:
        # First, check if tables exist
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Create users table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    created_at DATETIME NOT NULL
                )
            ''')
            
            # Create auth_users table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auth_users (
                    username TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    disabled BOOLEAN NOT NULL DEFAULT FALSE
                )
            ''')

            # Create interactions table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    user_id INTEGER NOT NULL,
                    article_id INTEGER NOT NULL,
                    interaction_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (article_id) REFERENCES articles (article_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    PRIMARY KEY (user_id, article_id, interaction_type)
                )
            ''')
            
            # Create article_boosts table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS article_boosts (
                    article_id INTEGER NOT NULL,
                    boost_factor REAL NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    boost_type TEXT NOT NULL,
                    FOREIGN KEY (article_id) REFERENCES articles (article_id)
                )
            ''')
            conn.commit()
            
        db = DBManager('recommender.db')
        
        # Load articles from CSV file
        print("Loading articles into database...")
        db.load_medium_articles('data/medium_articles_reduced.csv')
        
        # Verify articles were loaded
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM articles')
            article_count = cursor.fetchone()[0]
            print(f"\nSuccessfully loaded {article_count} articles")
            
            # Print first few articles for verification
            cursor.execute('SELECT article_id, title, tags FROM articles LIMIT 3')
            print("\nSample articles loaded:")
            for row in cursor.fetchall():
                print(f"ID: {row[0]}, Title: {row[1]}, Tags: {row[2]}")
        
        # Create sample users with interests
        print("\nCreating sample users...")
        sample_users = [
            {
                'username': 'tech_enthusiast',
                'email': 'tech@example.com',
                'password': 'password123',
                'interests': ['technology', 'programming', 'artificial intelligence']
            },
            {
                'username': 'data_scientist',
                'email': 'data@example.com',
                'password': 'password123',
                'interests': ['data science', 'machine learning', 'python']
            },
            {
                'username': 'business_pro',
                'email': 'business@example.com',
                'password': 'password123',
                'interests': ['startup', 'entrepreneurship', 'business']
            },
            {
                'username': 'creative_writer',
                'email': 'writer@example.com',
                'password': 'password123',
                'interests': ['writing', 'creativity', 'productivity']
            }
        ]
        
        # Create users and add interactions
        for user in sample_users:
            try:
                print(f"\nCreating user {user['username']}...")
                # Check if user exists before creating
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 FROM auth_users WHERE username = ?", (user['username'],))
                    if not cursor.fetchone():
                        # Create user in recommender system
                        db.create_user(user['username'])
                        
                        # Create user in auth system
                        hashed_password = pwd_context.hash(user['password'])
                        cursor.execute('''
                            INSERT INTO auth_users (username, email, hashed_password)
                            VALUES (?, ?, ?)
                        ''', (
                            user['username'],
                            user['email'],
                            hashed_password
                        ))
                        conn.commit()
                        print(f"Successfully created user {user['username']}")
                    else:
                        print(f"User {user['username']} already exists, skipping...")
                
            except Exception as e:
                print(f"Error processing user {user['username']}: {e}")
                continue
        
        print("\nInitial data loading complete!")
        print("\nSample users available:")
        for user in sample_users:
            print(f"Username: {user['username']}")
            print(f"Email: {user['email']}")
            print(f"Password: {user['password']}")
            print(f"Interests: {', '.join(user['interests'])}")
            print()
            
    except Exception as e:
        print(f"Error in data loading process: {e}")
        raise

if __name__ == "__main__":
    try:
        load_initial_data()
    except Exception as e:
        print(f"Failed to load initial data: {e}")