from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
import uvicorn
from db_manager import DBManager
from db_recommender import SQLiteRecommender, ArticleBoost
from jose import JWTError, jwt
from fastapi.middleware.cors import CORSMiddleware
import json
import sqlite3
import time
from passlib.context import CryptContext
import pandas as pd

app = FastAPI(
    title="TailorReads API",
    description="REST API for personalized article recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize database and recommender
DB_PATH = 'recommender.db'
db_manager = DBManager(DB_PATH)
recommender = SQLiteRecommender(DB_PATH)

# JWT Configuration
SECRET_KEY = "ankitnehasaad"  # In production, use a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

# Password hashing
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models for request/response validation
class UserCreate(BaseModel):
    username: str

class ArticleCreate(BaseModel):
    title: str
    text: str
    authors: List[str]
    tags: List[str]
    timestamp: Optional[datetime] = None

class InteractionCreate(BaseModel):
    username: str
    article_id: int
    interaction_type: str

class BoostCreate(BaseModel):
    article_id: int
    boost_factor: float
    start_time: datetime
    end_time: datetime
    boost_type: str

# Additional Pydantic models for auth
class UserSignup(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Helper functions for auth
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, email, disabled FROM auth_users WHERE username = ?",
            (token_data.username,)
        )
        user = cursor.fetchone()
        if user is None:
            raise credentials_exception
        if user[2]:  # disabled
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
    return user[0]  # return username

# User routes
@app.post("/users/", response_model=dict)
async def create_user(user: UserCreate):
    """Create a new user"""
    try:
        user_id = db_manager.create_user(user.username)
        return {"user_id": user_id, "username": user.username}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/")
async def list_users():
    """List all users and their interaction counts"""
    return db_manager.list_users()

@app.get("/users/{username}/interactions")
async def get_user_interactions(username: str):
    """Get interaction history for a user"""
    try:
        return db_manager.get_user_interactions(username)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Get a user's preference profile"""
    try:
        return recommender.get_user_profile(user_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/users/{username}/interests")
async def get_user_interests(
    username: str,
    current_user: str = Depends(get_current_user)
):
    """Get user interests"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT interests, updated_at
            FROM user_interests
            WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        if result:
            return {
                "interests": json.loads(result[0]),
                "updated_at": result[1]
            }
        return {"interests": [], "updated_at": None}

# Article routes
@app.post("/articles/")
async def create_article(article: ArticleCreate):
    """Add a new article"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get the next article_id
            cursor.execute("SELECT MAX(article_id) FROM articles")
            result = cursor.fetchone()
            next_article_id = (result[0] or 0) + 1
            
            # Insert the article with the new ID
            cursor.execute('''
                INSERT INTO articles (article_id, title, text, authors, tags, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                next_article_id,
                article.title,
                article.text,
                json.dumps(article.authors),
                json.dumps(article.tags),
                article.timestamp.isoformat() if article.timestamp else datetime.now().isoformat()
            ))
            conn.commit()
            
            return {"article_id": next_article_id}
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/articles/trending")
async def get_trending_articles(
    timeframe_hours: int = Query(24, gt=0),
    n_articles: int = Query(5, gt=0)
):
    """Get trending articles"""
    with get_db() as conn:
        # First try to get articles with interactions
        articles_df = pd.read_sql('''
            SELECT DISTINCT
                a.article_id,
                a.title,
                a.text,
                a.tags,
                a.timestamp,
                COUNT(i.article_id) as interaction_count
            FROM articles a
            LEFT JOIN interactions i ON a.article_id = i.article_id
            GROUP BY a.article_id
            ORDER BY interaction_count DESC, a.timestamp DESC
            LIMIT ?
        ''', conn, params=[n_articles])
        
        articles = []
        for _, article in articles_df.iterrows():
            # Clean up tags
            tags = json.loads(article['tags'])
            cleaned_tags = [tag.strip().strip("[]'\"") for tag in tags]
            cleaned_tags = [tag for tag in cleaned_tags if tag]
            
            articles.append({
                'article_id': int(article['article_id']),
                'title': article['title'],
                'text': article['text'][:200] + "..." if len(article['text']) > 200 else article['text'],
                'tags': cleaned_tags,
                'interaction_counts': {
                    'views': 0,
                    'likes': 0,
                    'shares': 0
                }
            })
        
        # Get interaction counts for these articles
        if articles:
            article_ids = [a['article_id'] for a in articles]
            placeholders = ','.join('?' * len(article_ids))
            interactions_df = pd.read_sql(f'''
                SELECT 
                    article_id,
                    interaction_type,
                    COUNT(*) as count
                FROM interactions
                WHERE article_id IN ({placeholders})
                GROUP BY article_id, interaction_type
            ''', conn, params=article_ids)
            
            # Update interaction counts
            for _, interaction in interactions_df.iterrows():
                article_id = interaction['article_id']
                interaction_type = interaction['interaction_type']
                count = interaction['count']
                
                for article in articles:
                    if article['article_id'] == article_id:
                        article['interaction_counts'][interaction_type + 's'] = int(count)
        
        return articles

@app.get("/articles/by-tag/{tag}")
async def get_articles_by_tag(
    tag: str,
    n_articles: int = Query(5, gt=0),
    include_boosted: bool = True
):
    """Get articles by tag"""
    articles = recommender.get_articles_by_tag(
        tag=tag,
        n_articles=n_articles,
        include_boosted=include_boosted
    )
    
    # Clean up tags
    for article in articles:
        if isinstance(article['tags'], str):
            article['tags'] = json.loads(article['tags'])
        article['tags'] = [tag.strip().strip("[]'\"") for tag in article['tags']]
        article['tags'] = [tag for tag in article['tags'] if tag]
    
    return articles

@app.get("/articles/{article_id}/similar")
async def get_similar_articles(
    article_id: int,
    n_recommendations: int = Query(5, gt=0),
    include_boosted: bool = True,
    similarity_threshold: float = Query(0.1, ge=0, le=1)
):
    """Get similar articles"""
    try:
        return recommender.get_similar_articles(
            article_id=article_id,
            n_recommendations=n_recommendations,
            include_boosted=include_boosted,
            similarity_threshold=similarity_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# Interaction routes
@app.post("/interactions/")
async def add_interaction(
    interaction: InteractionCreate,
    current_user: str = Depends(get_current_user)
):
    """Add a user interaction"""
    try:
        # Verify the user is interacting as themselves
        if interaction.username != current_user:
            raise HTTPException(
                status_code=403,
                detail="Cannot create interactions for other users"
            )
        
        # Validate interaction type
        valid_types = ['view', 'like', 'share', 'not_interested']
        if interaction.interaction_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interaction type. Must be one of: {valid_types}"
            )
            
        db_manager.add_interaction(
            username=interaction.username,
            article_id=interaction.article_id,
            interaction_type=interaction.interaction_type
        )
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation routes
@app.get("/recommendations/")
async def get_recommendations(
    user_id: Optional[int] = None,
    article_id: Optional[int] = None,
    n_recommendations: int = Query(5, gt=0),
    include_boosted: bool = True
):
    """Get recommendations based on user and/or article"""
    if user_id is None and article_id is None:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or article_id must be provided"
        )
    
    recommendations = recommender.get_recommendations(
        user_id=user_id,
        article_id=article_id,
        n_recommendations=n_recommendations,
        include_boosted=include_boosted
    )
    
    # Clean up tags in recommendations
    for rec in recommendations:
        tags = rec['tags']
        cleaned_tags = [tag.strip().strip("[]'\"") for tag in tags]
        cleaned_tags = [tag for tag in cleaned_tags if tag]
        rec['tags'] = cleaned_tags
    
    return recommendations

# Boost routes
@app.post("/boosts/")
async def add_article_boost(boost: BoostCreate):
    """Add a boost to an article"""
    try:
        recommender.add_article_boost(ArticleBoost(
            article_id=boost.article_id,
            boost_factor=boost.boost_factor,
            start_time=boost.start_time,
            end_time=boost.end_time,
            boost_type=boost.boost_type
        ))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Data loading route
@app.post("/load-articles/")
async def load_articles(csv_path: str):
    """Load articles from a CSV file"""
    try:
        db_manager.load_medium_articles(csv_path)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Auth routes
@app.post("/signup", response_model=Token)
async def signup(user: UserSignup):
    """Sign up a new user"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            # Check if username exists
            cursor.execute("SELECT 1 FROM auth_users WHERE username = ?", (user.username,))
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already registered"
                )
                
            # Check if email exists
            cursor.execute("SELECT 1 FROM auth_users WHERE email = ?", (user.email,))
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
                
            # Create new user in auth system
            hashed_password = get_password_hash(user.password)
            cursor.execute(
                "INSERT INTO auth_users (username, email, hashed_password) VALUES (?, ?, ?)",
                (user.username, user.email, hashed_password)
            )
            conn.commit()
            
            # Create user in recommender system
            try:
                cursor.execute('''
                    INSERT INTO users (username, created_at)
                    VALUES (?, ?)
                ''', (user.username, datetime.now().isoformat()))
                conn.commit()
            except Exception as e:
                # Rollback auth user if recommender user creation fails
                cursor.execute("DELETE FROM auth_users WHERE username = ?", (user.username,))
                conn.commit()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
            
            # Create access token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user.username}, expires_delta=access_token_expires
            )
            return {"access_token": access_token, "token_type": "bearer"}
            
        except sqlite3.Error as e:
            conn.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )

@app.post("/login", response_model=Token)
async def login_json(user: UserLogin):
    """Login with JSON request"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, hashed_password FROM auth_users WHERE username = ?",
            (user.username,)
        )
        user_data = cursor.fetchone()
        
        if not user_data or not verify_password(user.password, user_data[1]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_data[0]}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
async def login_form(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with form data (for OAuth2 compatibility)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, hashed_password FROM auth_users WHERE username = ?",
            (form_data.username,)
        )
        user = cursor.fetchone()
        
        if not user or not verify_password(form_data.password, user[1]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user[0]}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: str = Depends(get_current_user)):
    """Get current user information"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, email FROM auth_users WHERE username = ?",
            (current_user,)
        )
        user = cursor.fetchone()
        return {"username": user[0], "email": user[1]}

# Protect existing routes with authentication
# Example of protecting a route:
@app.get("/recommendations/", dependencies=[Depends(get_current_user)])
async def get_recommendations(
    user_id: Optional[int] = None,
    article_id: Optional[int] = None,
    n_recommendations: int = Query(5, gt=0),
    include_boosted: bool = True
):
    """Get recommendations based on user and/or article"""
    if user_id is None and article_id is None:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or article_id must be provided"
        )
    
    return recommender.get_recommendations(
        user_id=user_id,
        article_id=article_id,
        n_recommendations=n_recommendations,
        include_boosted=include_boosted
    )

# Add this new endpoint
@app.get("/articles/search")
async def search_articles(
    query: str,
    n_results: int = Query(10, gt=0),
    current_user: str = Depends(get_current_user)
):
    """Search articles by title, text, or tags"""
    with get_db() as conn:
        cursor = conn.cursor()
        search_term = f"%{query}%"
        cursor.execute('''
            SELECT DISTINCT 
                article_id, 
                title, 
                text, 
                authors, 
                tags, 
                timestamp
            FROM articles
            WHERE title LIKE ? 
               OR text LIKE ?
               OR tags LIKE ?
            LIMIT ?
        ''', (search_term, search_term, search_term, n_results))
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                "article_id": row[0],
                "title": row[1],
                "text": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                "authors": json.loads(row[3]),
                "tags": json.loads(row[4]),
                "timestamp": row[5]
            })
        
        return articles

# Add this endpoint for popular tags
@app.get("/articles/popular-tags")
async def get_popular_tags(
    n_tags: int = Query(10, gt=0),
    current_user: str = Depends(get_current_user)
):
    """Get most popular article tags"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT tags
            FROM articles
        ''')
        
        tag_counts = {}
        for (tags_json,) in cursor.fetchall():
            tags = json.loads(tags_json)
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort tags by frequency
        popular_tags = sorted(
            tag_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_tags]
        
        return [tag for tag, _ in popular_tags]

# Add this new endpoint to api.py
@app.get("/debug/status")
async def get_debug_status():
    """Get debug information about the database"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get article count
        cursor.execute("SELECT COUNT(*) FROM articles")
        article_count = cursor.fetchone()[0]
        
        # Get user count
        cursor.execute("SELECT COUNT(*) FROM auth_users")
        user_count = cursor.fetchone()[0]
        
        # Get interaction count
        cursor.execute("SELECT COUNT(*) FROM interactions")
        interaction_count = cursor.fetchone()[0]
        
        # Get sample articles
        cursor.execute("SELECT article_id, title, tags FROM articles LIMIT 5")
        sample_articles = [
            {
                "article_id": row[0],
                "title": row[1],
                "tags": json.loads(row[2])
            }
            for row in cursor.fetchall()
        ]
        
        return {
            "article_count": article_count,
            "user_count": user_count,
            "interaction_count": interaction_count,
            "sample_articles": sample_articles
        }

# Add this new endpoint
@app.get("/articles/{article_id}")
async def get_article(
    article_id: int,
    current_user: str = Depends(get_current_user)
):
    """Get article details"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT article_id, title, text, authors, tags, timestamp
            FROM articles
            WHERE article_id = ?
        ''', (article_id,))
        
        article = cursor.fetchone()
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        # Clean up tags
        tags = json.loads(article[4])
        cleaned_tags = [tag.strip().strip("[]'\"") for tag in tags]
        cleaned_tags = [tag for tag in cleaned_tags if tag]
            
        return {
            "article_id": article[0],
            "title": article[1],
            "text": article[2],
            "authors": json.loads(article[3]),
            "tags": cleaned_tags,  # Use cleaned tags
            "timestamp": article[5]
        }

# Add the get_db function
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

# Add this new model
class UserInterests(BaseModel):
    username: str
    interests: List[str]

# Add this new endpoint
@app.post("/users/interests")
async def save_user_interests(
    interests: UserInterests,
    current_user: str = Depends(get_current_user)
):
    """Save user interests"""
    if interests.username != current_user:
        raise HTTPException(
            status_code=403,
            detail="Cannot modify interests for other users"
        )
        
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            # Store interests in a new table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_interests (
                    username TEXT PRIMARY KEY,
                    interests TEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            ''')
            
            # Save or update interests
            cursor.execute('''
                INSERT OR REPLACE INTO user_interests (username, interests, updated_at)
                VALUES (?, ?, ?)
            ''', (
                interests.username,
                json.dumps(interests.interests),
                datetime.now().isoformat()
            ))
            conn.commit()
            return {"status": "success"}
            
        except sqlite3.Error as e:
            conn.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )

# Add these new endpoints
@app.post("/articles/boosts")
async def add_article_boost(
    boost: BoostCreate,
    current_user: str = Depends(get_current_user)
):
    """Add a boost to an article"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO article_boosts (
                    article_id, boost_factor, start_time, end_time, boost_type
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                boost.article_id,
                boost.boost_factor,
                boost.start_time,
                boost.end_time,
                boost.boost_type
            ))
            conn.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/articles/boosts")
async def get_article_boosts(
    current_user: str = Depends(get_current_user)
):
    """Get all current article boosts"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                b.article_id,
                a.title as article_title,
                b.boost_factor,
                b.start_time,
                b.end_time,
                b.boost_type
            FROM article_boosts b
            JOIN articles a ON b.article_id = a.article_id
            WHERE b.end_time > ?
            ORDER BY b.start_time DESC
        ''', (datetime.now().isoformat(),))
        
        boosts = []
        for row in cursor.fetchall():
            boosts.append({
                "article_id": row[0],
                "article_title": row[1],
                "boost_factor": row[2],
                "start_time": row[3],
                "end_time": row[4],
                "boost_type": row[5]
            })
        return boosts

@app.delete("/articles/boosts/{article_id}")
async def delete_article_boost(
    article_id: int,
    current_user: str = Depends(get_current_user)
):
    """Delete an article boost"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM article_boosts
                WHERE article_id = ?
            ''', (article_id,))
            conn.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 