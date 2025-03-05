import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Simulating social media sentiment analysis library
class SocialMediaSentimentAnalyzer:
    def __init__(self):
        self.sentiment_lexicon = {
            'awesome': 0.8, 'great': 0.7, 'good': 0.6, 
            'bad': -0.6, 'terrible': -0.8, 'poor': -0.7,
            'amazing': 0.9, 'excellent': 0.8, 'incredible': 0.9,
            'worst': -0.9, 'horrible': -0.8, 'disappointing': -0.7
        }
    
    def analyze_text(self, text):
        """
        Simple sentiment scoring based on word match
        """
        words = text.lower().split()
        sentiment_scores = [self.sentiment_lexicon.get(word, 0) for word in words]
        return np.mean(sentiment_scores) if sentiment_scores else 0

class SocialSalesSentimentAnalyzer:
    def __init__(self, sales_data_path, social_media_data_path):
        """
        Initialize analyzer with sales and social media data
        
        Args:
            sales_data_path (str): Path to sales data CSV
            social_media_data_path (str): Path to social media data CSV
        """
        self.sales_data = pd.read_csv(sales_data_path)
        self.social_media_data = pd.read_csv(social_media_data_path)
        self.sentiment_analyzer = SocialMediaSentimentAnalyzer()
        
    def preprocess_data(self):
        """
        Merge sales and social media data, compute sentiment scores
        """
        # Add sentiment score to social media data
        self.social_media_data['sentiment_score'] = self.social_media_data['post_text'].apply(
            self.sentiment_analyzer.analyze_text
        )
        
        # Aggregate social media sentiment by product
        social_sentiment = self.social_media_data.groupby('product_name').agg({
            'sentiment_score': ['mean', 'count'],
            'engagement': 'mean'
        }).reset_index()
        
        # Flatten multi-level column names
        social_sentiment.columns = [
            'product_name', 
            'avg_sentiment', 
            'social_media_mentions', 
            'avg_engagement'
        ]
        
        # Merge with sales data
        merged_data = pd.merge(
            self.sales_data, 
            social_sentiment, 
            on='product_name', 
            how='left'
        )
        
        # Fill missing values
        merged_data.fillna({
            'avg_sentiment': 0, 
            'social_media_mentions': 0, 
            'avg_engagement': 0
        }, inplace=True)
        
        return merged_data
    
    def feature_engineering(self, df):
        """
        Create advanced features combining sales and social data
        """
        # Extract time-based features
        df['month'] = pd.to_datetime(df['sale_date']).dt.month
        df['day_of_week'] = pd.to_datetime(df['sale_date']).dt.dayofweek
        
        # Create interaction features
        df['sentiment_sales_interaction'] = df['avg_sentiment'] * df['total_sales']
        df['engagement_sales_correlation'] = df['avg_engagement'] * df['total_sales']
        
        # One-hot encode categorical variables
        categorical_cols = ['product_category', 'product_name']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        return df_encoded
    
    def train_predictive_model(self, df):
        """
        Train machine learning model to predict sales
        """
        # Print dataset info for debugging
        print("\nDataset Information:")
        print(df.info())
        print("\nFeature Columns:")
        print(df.columns.tolist())
        
        # Prepare features and target
        features = [
            col for col in df.columns 
            if col not in ['total_sales', 'sale_date', 'product_name']
        ]
        
        # Ensure we have features
        if not features:
            raise ValueError("No features available for model training")
        
        X = df[features]
        y = df['total_sales']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with more trees and deeper learning
        model = RandomForestRegressor(
            n_estimators=200,  # Increased number of trees
            max_depth=10,      # Added max depth
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        
        print("\nSales Prediction Model Performance:")
        print(f"R-squared: {r2_score(y_test, y_pred):.2f}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        
        # Feature importance visualization
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Features Influencing Sales')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        return model, feature_importance
    
    def visualize_sentiment_sales_relationship(self, df):
        """
        Create visualizations showing sentiment and sales relationship
        """
        plt.figure(figsize=(12, 6))
        plt.scatter(df['avg_sentiment'], df['total_sales'], alpha=0.6)
        plt.title('Social Media Sentiment vs Sales')
        plt.xlabel('Average Sentiment Score')
        plt.ylabel('Total Sales')
        plt.tight_layout()
        plt.savefig('sentiment_sales_scatter.png')
        plt.close()