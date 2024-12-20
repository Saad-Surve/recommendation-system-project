import pandas as pd
import sqlite3
from datetime import datetime
import json
from db_recommender import SQLiteRecommender
from typing import List, Dict, Optional

class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # self._initialize_database()
        
    def _initialize_database(self):
        """Initialize the database with necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
        #     # Drop existing tables to avoid conflicts
        #     conn.execute('DROP TABLE IF EXISTS interactions')
        #     conn.execute('DROP TABLE IF EXISTS article_boosts')
        #     conn.execute('DROP TABLE IF EXISTS users')
        #     conn.execute('DROP TABLE IF EXISTS articles')
            
            # Create users table
            # conn.execute('''
            #     CREATE TABLE users (
            #         user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            #         username TEXT UNIQUE NOT NULL,
            #         created_at DATETIME NOT NULL
            #     )
            # ''')
            
            # # Create articles table
            # conn.execute('''
            #     CREATE TABLE articles (
            #         article_id INTEGER PRIMARY KEY,
            #         title TEXT NOT NULL,
            #         text TEXT NOT NULL,
            #         authors TEXT NOT NULL,
            #         tags TEXT NOT NULL,
            #         timestamp DATETIME NOT NULL
            #     )
            # ''')
            
            # Create interactions table
            conn.execute('''
                CREATE TABLE interactions (
                    user_id INTEGER NOT NULL,
                    article_id INTEGER NOT NULL,
                    interaction_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (article_id) REFERENCES articles (article_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    PRIMARY KEY (user_id, article_id, interaction_type)
                )
            ''')
            
            # Create article_boosts table
            conn.execute('''
                CREATE TABLE article_boosts (
                    article_id INTEGER NOT NULL,
                    boost_factor REAL NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    boost_type TEXT NOT NULL,
                    FOREIGN KEY (article_id) REFERENCES articles (article_id)
                )
            ''')
            
            # Commit changes
            conn.commit()
            
            # Other tables will be created by SQLiteRecommender
            recommender = SQLiteRecommender(self.db_path)
    
    def load_medium_articles(self, csv_path: str):
        """Load articles from the Medium dataset"""
        print(f"Loading articles from {csv_path}")
        
        # Read CSV and add article_id
        df = pd.read_csv(csv_path)
        df['article_id'] = range(len(df))  # Add article_id starting from 0
        
        # Process data for database
        df['authors'] = df['authors'].apply(lambda x: json.dumps([x]))
        df['tags'] = df['tags'].apply(lambda x: json.dumps(str(x).split(',')))
        
        # Handle timestamp parsing with a more flexible approach
        def parse_timestamp(ts):
            try:
                # Try parsing with pandas (handles multiple formats)
                return pd.to_datetime(ts).isoformat()
            except:
                # If parsing fails, return current timestamp
                return datetime.now().isoformat()
        
        df['timestamp'] = df['timestamp'].apply(parse_timestamp)
        
        # Select and rename columns
        articles_df = df[[
            'article_id',
            'title', 
            'text', 
            'authors',
            'tags', 
            'timestamp'
        ]].copy()
        
        # Clean text data
        articles_df['text'] = articles_df['text'].fillna('')
        articles_df['title'] = articles_df['title'].fillna('Untitled')
        
        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            # Use the article_id from the DataFrame instead of letting SQLite create one
            articles_df.to_sql('articles', conn, if_exists='replace', index=False)
            
        print(f"Loaded {len(articles_df)} articles into database")
    
    def create_user(self, username: str) -> int:
        """Create a new user and return their user_id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (username, created_at)
                    VALUES (?, ?)
                ''', (username, datetime.now().isoformat()))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError(f"Username '{username}' already exists")
    
    def get_user_id(self, username: str) -> Optional[int]:
        """Get user_id from username"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def add_interaction(
        self, 
        username: str, 
        article_id: int, 
        interaction_type: str,
        timestamp: Optional[datetime] = None
    ):
        """Add a user interaction"""
        user_id = self.get_user_id(username)
        if user_id is None:
            raise ValueError(f"User '{username}' not found")
            
        timestamp = timestamp or datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Verify article exists
            cursor.execute('SELECT 1 FROM articles WHERE article_id = ?', (article_id,))
            if not cursor.fetchone():
                raise ValueError(f"Article {article_id} not found")
            
            # Add interaction
            cursor.execute('''
                INSERT OR REPLACE INTO interactions 
                (user_id, article_id, interaction_type, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (user_id, article_id, interaction_type, timestamp.isoformat()))
    
    def get_user_interactions(self, username: str) -> List[Dict]:
        """Get all interactions for a user"""
        user_id = self.get_user_id(username)
        if user_id is None:
            raise ValueError(f"User '{username}' not found")
            
        with sqlite3.connect(self.db_path) as conn:
            interactions_df = pd.read_sql('''
                SELECT 
                    i.article_id,
                    a.title,
                    i.interaction_type,
                    i.timestamp
                FROM interactions i
                JOIN articles a ON i.article_id = a.article_id
                WHERE i.user_id = ?
                ORDER BY i.timestamp DESC
            ''', conn, params=[user_id])
            
        return interactions_df.to_dict('records')
    
    def list_users(self) -> List[Dict]:
        """List all users with their interaction counts"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql('''
                SELECT 
                    u.user_id,
                    u.username,
                    u.created_at,
                    COUNT(i.user_id) as interaction_count
                FROM users u
                LEFT JOIN interactions i ON u.user_id = i.user_id
                GROUP BY u.user_id, u.username, u.created_at
                ORDER BY u.user_id
            ''', conn).to_dict('records') 