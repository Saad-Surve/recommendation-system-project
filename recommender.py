import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

class InteractionTypes(Enum):
    VIEW = "view"
    LIKE = "like"
    SHARE = "share"
    NOT_INTERESTED = "not_interested"
    
    @classmethod
    def from_string(cls, string_value: str) -> 'InteractionTypes':
        """Convert string to InteractionTypes enum value"""
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
    boost_type: str  # 'sponsored', 'trending', 'editorial_pick'
    
class EnhancedHybridRecommender:
    def __init__(self, article_data: pd.DataFrame):
        """
        Initialize the enhanced hybrid recommender system
        
        Args:
            article_data: DataFrame with columns ['title', 'text', 'authors', 'timestamp', 'tags']
        """
        self.articles = article_data
        self.article_ids = list(range(len(article_data)))
        
        # Interaction weights for different types
        self.interaction_weights = {
            InteractionTypes.VIEW: 0.3,
            InteractionTypes.LIKE: 1.0,
            InteractionTypes.SHARE: 1.5,
            InteractionTypes.NOT_INTERESTED: -2.0
        }
        
        # Initialize matrices for different interaction types
        self.interaction_matrices = {
            interaction_type: csr_matrix((0, len(self.articles)))
            for interaction_type in InteractionTypes
        }
        
        # User preferences and article boosting
        self.user_preferences = defaultdict(lambda: defaultdict(float))
        self.article_boosts = []
        self.user_negative_feedback = defaultdict(set)
        
        # Content-based components
        self.content_similarities = None
        self.tfidf_vectorizer = None
        self.article_features = None
        self._prepare_content_features()
        
        # Collaborative filtering components
        self.collaborative_model = None
        self.user_embeddings = None
        self.article_embeddings = None
        
        # Initialize topic modeling
        self._initialize_topic_modeling()
        
    def _initialize_topic_modeling(self):
        """Initialize topic modeling for better content understanding"""
        # Using SVD for topic modeling
        self.topic_model = TruncatedSVD(n_components=50)
        self.topic_features = self.topic_model.fit_transform(self.article_features)
        
    def _prepare_content_features(self):
        """Prepare enhanced content features using TF-IDF and metadata"""
        # Combine text, tags, and authors with weights
        self.articles['processed_content'] = self.articles.apply(
            lambda x: (
                f"{x['text']} " + 
                f"{' '.join(x['tags'] * 3)} " +  # Increase tag importance
                f"{' '.join(x['authors'] * 2)}"  # Increase author importance
            ),
            axis=1
        )
        
        # Create TF-IDF features with enhanced parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        self.article_features = self.tfidf_vectorizer.fit_transform(
            self.articles['processed_content']
        )
        
        # Calculate content-based similarity matrix
        self.content_similarities = cosine_similarity(self.article_features)
        
    def add_article_boost(self, boost: ArticleBoost):
        """Add a boost to an article"""
        self.article_boosts.append(boost)
        # Clean expired boosts
        current_time = datetime.now()
        self.article_boosts = [
            b for b in self.article_boosts 
            if b.end_time > current_time
        ]
        
    def add_user_interaction(
        self, 
        user_id: int, 
        article_id: int, 
        interaction_type: str,
        timestamp: Optional[datetime] = None
    ):
        """
        Add a user interaction with type-specific handling
        
        Args:
            user_id: unique identifier for the user
            article_id: index of the article in self.articles
            interaction_type: type of interaction ('view', 'like', 'share', 'not_interested')
            timestamp: timestamp of the interaction
        """
        # Convert string interaction type to enum
        try:
            interaction_enum = InteractionTypes.from_string(interaction_type)
        except ValueError as e:
            raise ValueError(str(e))
            
        matrix = self.interaction_matrices[interaction_enum]
        
        # Expand matrix if needed
        if user_id >= matrix.shape[0]:
            new_rows = user_id - matrix.shape[0] + 1
            additional_matrix = csr_matrix((new_rows, matrix.shape[1]))
            self.interaction_matrices[interaction_enum] = vstack([matrix, additional_matrix])
            matrix = self.interaction_matrices[interaction_enum]
        
        # Update interaction matrix
        matrix[user_id, article_id] = 1
        
        # Update user preferences
        weight = self.interaction_weights[interaction_enum]
        self.user_preferences[user_id][article_id] += weight
        
        # Handle negative feedback
        if interaction_enum == InteractionTypes.NOT_INTERESTED:
            self.user_negative_feedback[user_id].add(article_id)
            # Add similar articles to negative feedback with reduced weight
            similar_articles = self.get_similar_articles(article_id, n_recommendations=5)
            for similar in similar_articles:
                self.user_preferences[user_id][similar['article_id']] += weight * 0.3
        
        # Update collaborative model if enough interactions
        if sum(matrix.getnnz() for matrix in self.interaction_matrices.values()) > 100:
            self._update_collaborative_model()
            
    def _update_collaborative_model(self):
        """Update the collaborative filtering model using weighted interactions"""
        # Combine interaction matrices with weights
        combined_matrix = csr_matrix(self.interaction_matrices[InteractionTypes.VIEW].shape)
        
        for int_type, matrix in self.interaction_matrices.items():
            weight = self.interaction_weights[int_type]
            combined_matrix += matrix * weight
        
        # Apply matrix factorization
        self.collaborative_model = TruncatedSVD(n_components=50)
        self.user_embeddings = self.collaborative_model.fit_transform(combined_matrix)
        self.article_embeddings = self.collaborative_model.components_.T
        
    def _get_article_boost_score(self, article_id: int) -> float:
        """Calculate current boost score for an article"""
        current_time = datetime.now()
        boost_score = 1.0
        
        for boost in self.article_boosts:
            if (boost.article_id == article_id and 
                boost.start_time <= current_time <= boost.end_time):
                boost_score *= boost.boost_factor
        
        return boost_score
    
    def get_recommendations(
        self, 
        user_id: Optional[int] = None, 
        article_id: Optional[int] = None,
        n_recommendations: int = 5,
        include_boosted: bool = True
    ) -> List[Dict]:
        """
        Get enhanced hybrid recommendations
        
        Args:
            user_id: ID of user to get recommendations for (optional)
            article_id: ID of article to get similar articles for (optional)
            n_recommendations: number of recommendations to return
            include_boosted: whether to include boosted articles
        """
        content_scores = np.zeros(len(self.articles))
        collaborative_scores = np.zeros(len(self.articles))
        
        # Get content-based scores
        if article_id is not None:
            content_scores = self.content_similarities[article_id]
            
        # Get collaborative filtering scores
        if user_id is not None and self.collaborative_model is not None:
            if user_id < self.user_embeddings.shape[0]:
                user_vector = self.user_embeddings[user_id]
                collaborative_scores = np.dot(user_vector, self.article_embeddings.T)
        
        # Combine scores with weights that depend on available data
        if self.collaborative_model is not None:
            # More weight to collaborative as we have more user data
            interaction_count = sum(matrix.getnnz() for matrix in self.interaction_matrices.values())
            collab_weight = min(0.8, interaction_count / 1000)  # Cap at 0.8
            content_weight = 1 - collab_weight
            final_scores = (content_weight * content_scores + 
                          collab_weight * collaborative_scores)
        else:
            final_scores = content_scores
        
        # Apply article boosting
        if include_boosted:
            boost_scores = np.array([
                self._get_article_boost_score(aid) 
                for aid in range(len(self.articles))
            ])
            final_scores *= boost_scores
        
        # Filter out articles with negative feedback
        if user_id is not None:
            negative_articles = self.user_negative_feedback[user_id]
            final_scores[list(negative_articles)] = float('-inf')
        
        # Get top recommendations
        recommended_ids = np.argsort(final_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in recommended_ids:
            recommendations.append({
                'article_id': idx,
                'title': self.articles.iloc[idx]['title'],
                'score': float(final_scores[idx]),
                'tags': self.articles.iloc[idx]['tags'],
                'boost_score': float(self._get_article_boost_score(idx)),
                'is_boosted': self._get_article_boost_score(idx) > 1.0
            })
        
        return recommendations
    
    def get_user_profile(self, user_id: int) -> Dict:
        """Get a user's preference profile"""
        # Check if user_id is out of range for the interaction matrices
        max_user_id = max(matrix.shape[0] for matrix in self.interaction_matrices.values())
        if user_id >= max_user_id:
            return {
                'total_interactions': 0,
                'favorite_tags': [],
                'interaction_counts': {int_type.name: 0 for int_type in InteractionTypes}
            }
                
        # Count interactions by type safely
        interaction_counts = {}
        for int_type in InteractionTypes:
            matrix = self.interaction_matrices[int_type]
            if user_id < matrix.shape[0]:
                # Get the user's row as a sparse matrix
                user_row = matrix[user_id]
                interaction_counts[int_type.name] = user_row.getnnz()
            else:
                interaction_counts[int_type.name] = 0
            
        # Get favorite tags from positive interactions
        user_articles = [
            aid for aid, score in self.user_preferences[user_id].items()
            if score > 0
        ]
        
        tag_counts = defaultdict(int)
        for aid in user_articles:
            if aid < len(self.articles):
                tags = self.articles.iloc[aid]['tags']
                # Handle both string and list representations of tags
                if isinstance(tags, str):
                    try:
                        tags = eval(tags)  # Safely convert string representation to list
                    except:
                        tags = [tags]
                for tag in tags:
                    tag_counts[tag] += 1
                    
        favorite_tags = sorted(
            tag_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'total_interactions': sum(interaction_counts.values()),
            'favorite_tags': [tag for tag, _ in favorite_tags],
            'interaction_counts': interaction_counts
        }
    
    def get_trending_articles(self, timeframe_hours: int = 24, n_articles: int = 5) -> List[Dict]:
        """Get trending articles based on recent interactions"""
        current_time = datetime.now()
        
        # Initialize trending scores as numpy array
        trending_scores = np.zeros(len(self.articles))
        
        # Calculate trending scores based on recent interactions
        for int_type in InteractionTypes:
            if int_type != InteractionTypes.NOT_INTERESTED:
                matrix = self.interaction_matrices[int_type]
                weight = self.interaction_weights[int_type]
                
                # Sum up weighted interactions and ensure numpy array format
                article_interactions = np.array(matrix.sum(axis=0)).flatten()
                trending_scores += article_interactions * weight
        
        # Get top trending articles
        trending_ids = np.argsort(trending_scores)[::-1][:n_articles]
        
        trending_articles = []
        for idx in trending_ids:
            trending_articles.append({
                'article_id': int(idx),
                'title': self.articles.iloc[idx]['title'],
                'trending_score': float(trending_scores[idx]),
                'tags': self.articles.iloc[idx]['tags']
            })
            
        return trending_articles
    def get_similar_articles(
        self, 
        article_id: int, 
        n_recommendations: int = 5,
        include_boosted: bool = True,
        similarity_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Get similar articles based on content and interaction patterns
        
        Args:
            article_id: ID of the article to find similarities for
            n_recommendations: number of similar articles to return
            include_boosted: whether to include boosted articles
            similarity_threshold: minimum similarity score to include
            
        Returns:
            List of similar articles with similarity details
        """
        # Get content-based similarity scores
        content_similarities = self.content_similarities[article_id]
        
        # Get topic-based similarity scores
        topic_similarities = cosine_similarity(
            self.topic_features[article_id:article_id+1], 
            self.topic_features
        )[0]
        
        # Get interaction-based similarity if we have collaborative data
        interaction_similarities = np.zeros(len(self.articles))
        if self.collaborative_model is not None:
            article_vector = self.article_embeddings[article_id]
            interaction_similarities = np.dot(self.article_embeddings, article_vector)
        
        # Combine similarity scores
        final_similarities = (
            0.4 * content_similarities +
            0.4 * topic_similarities +
            0.2 * interaction_similarities
        )
        
        # Apply boosting if requested
        if include_boosted:
            boost_scores = np.array([
                self._get_article_boost_score(aid) 
                for aid in range(len(self.articles))
            ])
            final_similarities *= boost_scores
        
        # Filter by threshold and get top articles
        final_similarities[final_similarities < similarity_threshold] = 0
        final_similarities[article_id] = 0  # Exclude the input article
        similar_ids = np.argsort(final_similarities)[::-1][:n_recommendations]
        
        similar_articles = []
        for idx in similar_ids:
            if final_similarities[idx] > 0:
                similar_articles.append({
                    'article_id': idx,
                    'title': self.articles.iloc[idx]['title'],
                    'overall_similarity': float(final_similarities[idx]),
                    'content_similarity': float(content_similarities[idx]),
                    'topic_similarity': float(topic_similarities[idx]),
                    'interaction_similarity': float(interaction_similarities[idx]),
                    'tags': self.articles.iloc[idx]['tags'],
                    'shared_tags': list(
                        set(self.articles.iloc[idx]['tags']) & 
                        set(self.articles.iloc[article_id]['tags'])
                    ),
                    'boost_score': float(self._get_article_boost_score(idx)),
                    'is_boosted': self._get_article_boost_score(idx) > 1.0
                })
        
        return similar_articles


    def get_articles_by_tag(
        self,
        tag: str,
        n_articles: int = 5,
        include_boosted: bool = True
    ) -> List[Dict]:
        """Get articles by tag with smart ranking"""
        matching_articles = []
        for idx, article in self.articles.iterrows():
            if tag in article['tags']:
                score = 1.0
                
                # Parse timestamp from string if needed
                try:
                    article_timestamp = article['timestamp']
                    if isinstance(article_timestamp, str):
                        article_timestamp = datetime.fromisoformat(article_timestamp.replace('Z', '+00:00'))
                    
                    # Factor in recency 
                    days_old = (datetime.now(article_timestamp.tzinfo) - article_timestamp).days
                    recency_score = 1.0 / (1.0 + days_old/30)  # Decay over 30 days
                    score *= recency_score
                except (ValueError, TypeError, AttributeError):
                    # If timestamp parsing fails, use neutral recency score
                    score *= 1.0
                
                # Factor in interaction counts if available
                if self.collaborative_model is not None:
                    interaction_score = sum(
                        matrix[:, idx].sum() * weight
                        for matrix, weight in zip(
                            self.interaction_matrices.values(),
                            self.interaction_weights.values()
                        )
                    )
                    score *= (1.0 + interaction_score)
                
                # Apply boosting if requested
                boost_score = 1.0
                if include_boosted:
                    boost_score = self._get_article_boost_score(idx)
                    score *= boost_score
                
                matching_articles.append({
                    'article_id': idx,
                    'title': article['title'],
                    'tags': article['tags'],
                    'score': float(score),
                    'timestamp': article['timestamp'],
                    'is_boosted': boost_score > 1.0 if include_boosted else False
                })
        
        # Sort by score and return top n
        matching_articles.sort(key=lambda x: x['score'], reverse=True)
        return matching_articles[:n_articles]
