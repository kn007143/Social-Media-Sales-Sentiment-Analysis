from social_sales_sentiment import SocialSalesSentimentAnalyzer

def main():
    # Paths to your data files
    sales_path = 'data/sales_data.csv'
    social_media_path = 'data/social_media_data.csv'
    
    # Initialize analyzer
    analyzer = SocialSalesSentimentAnalyzer(sales_path, social_media_path)
    
    # Preprocess data
    processed_data = analyzer.preprocess_data()
    
    # Feature engineering
    engineered_data = analyzer.feature_engineering(processed_data)
    
    # Train predictive model
    model, feature_importance = analyzer.train_predictive_model(engineered_data)
    
    # Visualize sentiment-sales relationship
    analyzer.visualize_sentiment_sales_relationship(processed_data)

if __name__ == '__main__':
    main()