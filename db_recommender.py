import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd

class InteractionTypes(Enum):
    VIEW = "view"
    LIKE = "like"
    SHARE = "share"
    NOT_INTERESTED = "not_interested"
    
    @classmethod
    def from_string(cls, string_value: str) -> 'InteractionTypes':
        try:
            return cls(string_value.lower())
        except ValueError:
            raise ValueError(f"Invalid interaction type. Must be one of: {[e.value for e in cls]}")

@dataclass
class ArticleBoost:
    article_id: int
    boost_factor: float
    start_time: datetime
    end_time: datetime
    boost_type: str

class SQLiteRecommender:
    def __init__(self, db_path: str):
        """
        Initialize the SQLite-based recommender system
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.interaction_weights = {
            InteractionTypes.VIEW: 0.3,
            InteractionTypes.LIKE: 1.0,
            InteractionTypes.SHARE: 1.5,
            InteractionTypes.NOT_INTERESTED: -2.0
        }
        
        # Content-based components
        self.content_similarities = None
        self.tfidf_vectorizer = None
        self.article_features = None
        self.topic_model = None
        self.topic_features = None
        
    def _prepare_content_features(self):
        """Prepare content features for articles"""
        if self.content_similarities is not None:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            # Check if articles exist
            count = pd.read_sql('SELECT COUNT(*) as count FROM articles', conn)['count'].iloc[0]
            if count == 0:
                return
                
            # Get article data
            df = pd.read_sql('''
                SELECT article_id, title, text, authors, tags
                FROM articles
            ''', conn)
            
            # Process content
            df['processed_content'] = df.apply(
                lambda x: (
                    f"{x['text']} " + 
                    f"{' '.join(json.loads(x['tags']))} " +
                    f"{' '.join(json.loads(x['authors']))}"
                ),
                axis=1
            )
            
            # Create TF-IDF features
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.article_features = self.tfidf_vectorizer.fit_transform(
                df['processed_content']
            )
            
            # Calculate similarities
            self.content_similarities = cosine_similarity(self.article_features)
            
            # Initialize topic modeling
            self.topic_model = TruncatedSVD(n_components=50)
            self.topic_features = self.topic_model.fit_transform(self.article_features)
    
    def add_article(self, title: str, text: str, authors: List[str], 
                   tags: List[str], timestamp: datetime) -> int:
        """Add a new article to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO articles (title, text, authors, tags, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                title,
                text,
                json.dumps(authors),
                json.dumps(tags),
                timestamp.isoformat()
            ))
            article_id = cursor.lastrowid
            
        # Reset content features to force recalculation
        self.content_similarities = None
        self.tfidf_vectorizer = None
        self.article_features = None
        self.topic_model = None
        self.topic_features = None
        
        return article_id

    def add_user_interaction(self, user_id: int, article_id: int, 
                           interaction_type: str, timestamp: Optional[datetime] = None):
        """Add a user interaction to the database"""
        interaction_enum = InteractionTypes.from_string(interaction_type)
        timestamp = timestamp or datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO interactions 
                (user_id, article_id, interaction_type, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (
                user_id,
                article_id,
                interaction_enum.value,
                timestamp.isoformat()
            ))

    def add_article_boost(self, boost: ArticleBoost):
        """Add a boost to an article"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO article_boosts 
                (article_id, boost_factor, start_time, end_time, boost_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                boost.article_id,
                boost.boost_factor,
                boost.start_time.isoformat(),
                boost.end_time.isoformat(),
                boost.boost_type
            ))

    def get_recommendations(
        self,
        user_id: Optional[int] = None,
        article_id: Optional[int] = None,
        n_recommendations: int = 5,
        include_boosted: bool = True
    ) -> List[Dict]:
        """Get recommendations based on user history, interests, and/or article similarity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, get all articles and their current boosts
                articles_df = pd.read_sql('''
                    SELECT DISTINCT
                        a.article_id,
                        a.title,
                        a.text,
                        a.tags,
                        a.timestamp,
                        COUNT(i.article_id) as interaction_count,
                        COALESCE(b.boost_factor, 1.0) as boost_factor
                    FROM articles a
                    LEFT JOIN interactions i ON a.article_id = i.article_id
                    LEFT JOIN (
                        SELECT article_id, boost_factor
                        FROM article_boosts
                        WHERE datetime('now') BETWEEN start_time AND end_time
                    ) b ON a.article_id = b.article_id
                    GROUP BY a.article_id
                    ORDER BY boost_factor DESC NULLS LAST, interaction_count DESC, a.timestamp DESC
                ''', conn)
                
                if len(articles_df) == 0:
                    return []
                
                # Initialize scores with boost factors
                scores = np.ones(len(articles_df))
                if include_boosted:
                    print("Applying boosts...")
                    boost_factors = articles_df['boost_factor']
                    print(f"Boost factors found: {boost_factors.value_counts().to_dict()}")
                    scores *= boost_factors.values
                    
                    # Double check active boosts
                    current_time = datetime.now().isoformat()
                    boosts_df = pd.read_sql('''
                        SELECT article_id, boost_factor 
                        FROM article_boosts
                    ''', conn)
                    print(boosts_df)
                    
                    if not boosts_df.empty:
                        print(f"Active boosts found: {boosts_df.to_dict('records')}")
                        for _, boost in boosts_df.iterrows():
                            boost_mask = articles_df['article_id'] == boost['article_id']
                            if boost_mask.any():
                                scores[boost_mask] *= float(boost['boost_factor'])
                
                # If no criteria provided, return most recent articles
                if user_id is None and article_id is None:
                    articles_df['timestamp'] = pd.to_datetime(articles_df['timestamp'])
                    articles_df = articles_df.sort_values('timestamp', ascending=False)
                    recent_articles = []
                    for _, article in articles_df.head(n_recommendations).iterrows():
                        try:
                            tags = json.loads(article['tags'])
                            if isinstance(tags, str):
                                tags = json.loads(tags)
                            cleaned_tags = [tag.strip().strip("[]'\"") for tag in tags]
                            cleaned_tags = [tag for tag in cleaned_tags if tag]
                            
                            recent_articles.append({
                                'article_id': int(article['article_id']),
                                'title': str(article['title']),
                                'text': str(article['text']),
                                'score': 1.0,
                                'tags': cleaned_tags,
                            })
                        except Exception as e:
                            print(f"Error processing article {article['article_id']}: {str(e)}")
                            continue
                    return recent_articles
                
                # Add interest-based scores
                if user_id is not None:
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT username FROM users WHERE user_id = ?
                        ''', (user_id,))
                        user_result = cursor.fetchone()
                        
                        if user_result:
                            username = user_result[0]
                            cursor.execute('''
                                SELECT interests FROM user_interests
                                WHERE username = ?
                            ''', (username,))
                            interests_result = cursor.fetchone()
                            
                            if interests_result and interests_result[0]:
                                interests = json.loads(interests_result[0])
                                for idx, article in articles_df.iterrows():
                                    try:
                                        article_tags = json.loads(article['tags'])
                                        if isinstance(article_tags, str):
                                            article_tags = json.loads(article_tags)
                                        matching_interests = set(interests) & set(article_tags)
                                        if matching_interests:
                                            scores[idx] += 0.5 * len(matching_interests)
                                    except:
                                        continue
                    except Exception as e:
                        print(f"Error processing user interests: {str(e)}")
                
                # Add content-based scores
                if article_id is not None:
                    try:
                        article_idx = articles_df.index[articles_df['article_id'] == article_id].tolist()
                        if article_idx:
                            article_idx = article_idx[0]
                            article_row = articles_df.iloc[article_idx]
                            
                            # Calculate text similarity
                            texts = [f"{row['title']} {row['text']}" for _, row in articles_df.iterrows()]
                            tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
                            tfidf_matrix = tfidf.fit_transform(texts)
                            similarities = cosine_similarity(
                                tfidf_matrix[article_idx:article_idx+1],
                                tfidf_matrix
                            )[0]
                            scores += similarities
                            
                            # Add tag similarity
                            try:
                                article_tags = json.loads(article_row['tags'])
                                if isinstance(article_tags, str):
                                    article_tags = json.loads(article_tags)
                                for idx, row in articles_df.iterrows():
                                    if idx != article_idx:
                                        row_tags = json.loads(row['tags'])
                                        if isinstance(row_tags, str):
                                            row_tags = json.loads(row_tags)
                                        matching_tags = set(article_tags) & set(row_tags)
                                        if matching_tags:
                                            scores[idx] += 0.3 * len(matching_tags)
                            except:
                                pass
                    except Exception as e:
                        print(f"Error in content-based scoring: {str(e)}")
                
                # Apply boosts
                if include_boosted:
                    print("in boosted")
                    try:
                        current_time = datetime.now().isoformat()
                        boosts_df = pd.read_sql('''
                            SELECT article_id, boost_factor 
                            FROM article_boosts
                            WHERE start_time <= ? AND end_time >= ?
                        ''', conn, params=[current_time, current_time])
                        
                        for _, boost in boosts_df.iterrows():
                            boost_mask = articles_df['article_id'] == boost['article_id']
                            if boost_mask.any():
                                scores[boost_mask] *= float(boost['boost_factor'])
                    except Exception as e:
                        print(f"Error applying boosts: {str(e)}")
                
                # When getting final recommendations, multiply by boost factor again
                # to give extra weight to boosted articles
                if include_boosted:
                    boost_multiplier = articles_df['boost_factor'].fillna(1.0)
                    scores *= boost_multiplier.values
                
                # Get recommendations
                recommendations = []
                top_indices = np.argsort(scores)[::-1]
                
                for idx in top_indices:
                    if len(recommendations) >= n_recommendations:
                        break
                        
                    try:
                        article = articles_df.iloc[idx]
                        if article_id is not None and article['article_id'] == article_id:
                            continue
                            
                        tags = json.loads(article['tags'])
                        if isinstance(tags, str):
                            tags = json.loads(tags)
                        cleaned_tags = [tag.strip().strip("[]'\"") for tag in tags]
                        cleaned_tags = [tag for tag in cleaned_tags if tag]
                        
                        recommendations.append({
                            'article_id': int(article['article_id']),
                            'title': str(article['title']),
                            'text': str(article['text']),
                            'score': float(scores[idx]),
                            'tags': cleaned_tags,
                            'boosted': bool(article['boost_factor'] and article['boost_factor'] > 1.0)
                        })
                    except Exception as e:
                        print(f"Error processing recommendation {idx}: {str(e)}")
                        continue
                
                return recommendations
                
        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            return []

    def get_similar_articles(
        self, 
        article_id: int, 
        n_recommendations: int = 5,
        include_boosted: bool = True,
        similarity_threshold: float = 0.1
    ) -> List[Dict]:
        """Get similar articles based on content and interaction patterns"""
        with sqlite3.connect(self.db_path) as conn:
            # Get article info
            article_df = pd.read_sql('''
                SELECT article_id, title, tags 
                FROM articles 
                WHERE article_id = ?
            ''', conn, params=[article_id])
            
            if len(article_df) == 0:
                return []
            
            # Get all articles for comparison
            all_articles_df = pd.read_sql('''
                SELECT article_id, title, tags 
                FROM articles
            ''', conn)
            
            # Calculate similarities
            content_similarities = self.content_similarities[article_id]
            topic_similarities = cosine_similarity(
                self.topic_features[article_id:article_id+1], 
                self.topic_features
            )[0]
            
            # Get interaction-based similarity
            interaction_similarities = np.zeros(len(all_articles_df))
            interaction_counts = pd.read_sql('''
                SELECT a1.article_id as article1, a2.article_id as article2, COUNT(*) as common_users
                FROM interactions a1
                JOIN interactions a2 ON a1.user_id = a2.user_id
                WHERE a1.article_id = ?
                GROUP BY a2.article_id
            ''', conn, params=[article_id])
            
            if len(interaction_counts) > 0:
                for _, row in interaction_counts.iterrows():
                    interaction_similarities[row['article2']] = row['common_users']
                # Normalize
                if interaction_similarities.max() > 0:
                    interaction_similarities = interaction_similarities / interaction_similarities.max()
            
            # Combine similarities
            final_similarities = (
                0.4 * content_similarities +
                0.4 * topic_similarities +
                0.2 * interaction_similarities
            )
            
            # Apply boosting if requested
            if include_boosted:
                current_time = datetime.now().isoformat()
                boosts_df = pd.read_sql('''
                    SELECT article_id, boost_factor 
                    FROM article_boosts
                    WHERE start_time <= ? AND end_time >= ?
                ''', conn, params=[current_time, current_time])
                
                for _, boost in boosts_df.iterrows():
                    final_similarities[boost['article_id']] *= boost['boost_factor']
            
            # Filter and sort
            final_similarities[final_similarities < similarity_threshold] = 0
            final_similarities[article_id] = 0  # Exclude input article
            similar_indices = np.argsort(final_similarities)[::-1][:n_recommendations]
            
            similar_articles = []
            article_tags = json.loads(article_df.iloc[0]['tags'])
            
            for idx in similar_indices:
                if final_similarities[idx] > 0:
                    article = all_articles_df.iloc[idx]
                    article_tags_current = json.loads(article['tags'])
                    similar_articles.append({
                        'article_id': int(article['article_id']),
                        'title': article['title'],
                        'overall_similarity': float(final_similarities[idx]),
                        'content_similarity': float(content_similarities[idx]),
                        'topic_similarity': float(topic_similarities[idx]),
                        'interaction_similarity': float(interaction_similarities[idx]),
                        'tags': article_tags_current,
                        'shared_tags': list(set(article_tags_current) & set(article_tags))
                    })
            
            return similar_articles

    def get_trending_articles(self, timeframe_hours: int = 24, n_articles: int = 5) -> List[Dict]:
        """Get trending articles based on recent interactions"""
        with sqlite3.connect(self.db_path) as conn:
            # Calculate time threshold
            time_threshold = (
                datetime.now() - pd.Timedelta(hours=timeframe_hours)
            ).isoformat()
            
            # Get weighted interaction counts
            trending_df = pd.read_sql('''
                SELECT 
                    a.article_id,
                    a.title,
                    a.tags,
                    COUNT(CASE WHEN i.interaction_type = 'view' THEN 1 END) as views,
                    COUNT(CASE WHEN i.interaction_type = 'like' THEN 1 END) as likes,
                    COUNT(CASE WHEN i.interaction_type = 'share' THEN 1 END) as shares
                FROM articles a
                LEFT JOIN interactions i ON a.article_id = i.article_id
                WHERE i.timestamp >= ?
                GROUP BY a.article_id, a.title, a.tags
                ORDER BY (
                    views * ? + 
                    likes * ? + 
                    shares * ?
                ) DESC
                LIMIT ?
            ''', conn, params=[
                time_threshold,
                self.interaction_weights[InteractionTypes.VIEW],
                self.interaction_weights[InteractionTypes.LIKE],
                self.interaction_weights[InteractionTypes.SHARE],
                n_articles
            ])
            
            trending_articles = []
            for _, article in trending_df.iterrows():
                trending_score = (
                    article['views'] * self.interaction_weights[InteractionTypes.VIEW] +
                    article['likes'] * self.interaction_weights[InteractionTypes.LIKE] +
                    article['shares'] * self.interaction_weights[InteractionTypes.SHARE]
                )
                trending_articles.append({
                    'article_id': int(article['article_id']),
                    'title': article['title'],
                    'trending_score': float(trending_score),
                    'tags': json.loads(article['tags']),
                    'interaction_counts': {
                        'views': int(article['views']),
                        'likes': int(article['likes']),
                        'shares': int(article['shares'])
                    }
                })
            
            return trending_articles

    def get_user_profile(self, user_id: int) -> Dict:
        """Get a user's preference profile"""
        with sqlite3.connect(self.db_path) as conn:
            # Get interaction counts by type
            interaction_counts = pd.read_sql('''
                SELECT interaction_type, COUNT(*) as count
                FROM interactions
                WHERE user_id = ?
                GROUP BY interaction_type
            ''', conn, params=[user_id])
            
            # Get articles with positive interactions
            positive_articles = pd.read_sql('''
                SELECT DISTINCT a.article_id, a.tags
                FROM interactions i
                JOIN articles a ON i.article_id = a.article_id
                WHERE i.user_id = ? 
                AND i.interaction_type != 'not_interested'
            ''', conn, params=[user_id])
            
            # Process interaction counts
            counts_dict = {
                int_type.name: 0 for int_type in InteractionTypes
            }
            for _, row in interaction_counts.iterrows():
                counts_dict[InteractionTypes.from_string(row['interaction_type']).name] = int(row['count'])
            
            # Process favorite tags
            tag_counts = {}
            for _, article in positive_articles.iterrows():
                tags = json.loads(article['tags'])
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            favorite_tags = sorted(
                tag_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                'total_interactions': sum(counts_dict.values()),
                'favorite_tags': [tag for tag, _ in favorite_tags],
                'interaction_counts': counts_dict
            }

    def get_articles_by_tag(
        self,
        tag: str,
        n_articles: int = 5,
        include_boosted: bool = True
    ) -> List[Dict]:
        """Get articles by tag with smart ranking"""
        with sqlite3.connect(self.db_path) as conn:
            # Get articles with the specified tag
            articles_df = pd.read_sql('''
                SELECT 
                    a.article_id,
                    a.title,
                    a.tags,
                    a.timestamp,
                    COUNT(i.article_id) as interaction_count
                FROM articles a
                LEFT JOIN interactions i ON a.article_id = i.article_id
                WHERE json_extract(a.tags, '$') LIKE ?
                GROUP BY a.article_id, a.title, a.tags, a.timestamp
            ''', conn, params=[f'%{tag}%'])
            
            if len(articles_df) == 0:
                return []
            
            # Calculate scores
            scores = np.zeros(len(articles_df))
            current_time = datetime.now(timezone.utc)
            
            for idx, article in articles_df.iterrows():
                # Base score
                scores[idx] = 1.0
                
                # Recency score
                article_time = datetime.fromisoformat(article['timestamp'].replace('Z', '+00:00'))
                if article_time.tzinfo is None:
                    article_time = article_time.replace(tzinfo=timezone.utc)
                
                days_old = (current_time - article_time).days
                recency_score = 1.0 / (1.0 + days_old/30)
                scores[idx] *= recency_score
                
                # Interaction score
                if article['interaction_count'] > 0:
                    scores[idx] *= (1.0 + np.log1p(article['interaction_count']))
            
            # Apply boosting if requested
            if include_boosted:
                current_time_str = current_time.isoformat()
                boosts_df = pd.read_sql('''
                    SELECT article_id, boost_factor 
                    FROM article_boosts
                    WHERE start_time <= ? AND end_time >= ?
                ''', conn, params=[current_time_str, current_time_str])
                
                for _, boost in boosts_df.iterrows():
                    article_idx = articles_df['article_id'] == boost['article_id']
                    scores[article_idx] *= boost['boost_factor']
            
            # Sort and return top articles
            top_indices = np.argsort(scores)[::-1][:n_articles]
            
            results = []
            for idx in top_indices:
                article = articles_df.iloc[idx]
                results.append({
                    'article_id': int(article['article_id']),
                    'title': article['title'],
                    'tags': json.loads(article['tags']),
                    'score': float(scores[idx]),
                    'timestamp': article['timestamp'],
                    'interaction_count': int(article['interaction_count'])
                })
            
            return results

