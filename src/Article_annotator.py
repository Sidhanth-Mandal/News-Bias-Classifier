import pandas as pd
import json
import time
import asyncio
import aiohttp
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bias_annotation.log'),
        logging.StreamHandler()
    ]
)

class GeminiBiasAnnotator:
    def __init__(self, api_keys: List[str]):
        """
        Initialize the bias annotator with multiple API keys for rate limiting
        
        Args:
            api_keys: List of Gemini API keys
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.requests_per_key = []
        self.last_request_time = []
        
        # Initialize tracking for each API key
        for _ in api_keys:
            self.requests_per_key.append(0)
            self.last_request_time.append(datetime.now() - timedelta(minutes=1))
    
    def get_system_prompt(self) -> str:
        """
        System prompt to set the model's role for political bias annotation
        """
        return """You are a political bias analysis expert specializing in Indian news media. Your task is to analyze news articles and determine their political bias on a left-center-right spectrum specific to the Indian political context.

CRITICAL INSTRUCTIONS:
1. You MUST respond with ONLY valid JSON - no explanations, no markdown, no extra text
2. Analyze each article for political bias in the Indian context (BJP/right vs Congress/left vs neutral)
3. Assign bias scores that sum to approximately 1.0
4. Be conservative - if genuinely neutral, assign high center score
5. Consider: word choice, framing, source selection, emphasis, and tone
6. Your response must be a JSON array with exactly the same number of objects as input articles"""

    def get_user_prompt_template(self) -> str:
        """
        User prompt template where articles will be inserted
        """
        return """Analyze these Indian news articles for political bias. Return ONLY a JSON array with this exact structure:

[
  {
    "article_id": 0,
    "bias_scores": {
      "left": 0.0,
      "center": 0.0,
      "right": 0.0
    },
    "explanation": "Brief explanation in 1-2 sentences"
  }
]

Articles to analyze:
{articles_json}

Remember: 
- JSON ONLY response
- bias_scores must sum to ~1.0
- Use Indian political context (BJP=right, Congress=left)
- Be conservative with bias detection"""

    def prepare_articles_for_prompt(self, articles_batch: List[Dict]) -> str:
        """
        Convert a batch of articles to JSON format for the prompt
        """
        formatted_articles = []
        for i, article in enumerate(articles_batch):
            formatted_articles.append({
                "id": i,
                "title": article.get('title', ''),
                "text": article.get('text', '')[:3000]  # Limit text length to avoid token limits
            })
        return json.dumps(formatted_articles, indent=2, ensure_ascii=False)

    def get_next_api_key(self) -> str:
        """
        Get the next API key in rotation with rate limiting
        """
        current_time = datetime.now()
        
        # Check if we need to wait before using any key
        for i in range(len(self.api_keys)):
            time_since_last = (current_time - self.last_request_time[i]).total_seconds()
            if time_since_last >= 5:  # 12 requests per minute = 5 seconds between requests
                self.current_key_index = i
                self.last_request_time[i] = current_time
                self.requests_per_key[i] += 1
                return self.api_keys[i]
        
        # If all keys are on cooldown, wait for the earliest available
        min_wait_time = min([
            5 - (current_time - last_time).total_seconds() 
            for last_time in self.last_request_time
        ])
        
        if min_wait_time > 0:
            time.sleep(min_wait_time + 0.1)  # Small buffer
            return self.get_next_api_key()
        
        # Fallback to round-robin
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.last_request_time[self.current_key_index] = current_time
        self.requests_per_key[self.current_key_index] += 1
        return self.api_keys[self.current_key_index]

    async def call_gemini_api(self, session: aiohttp.ClientSession, articles_batch: List[Dict], batch_id: int) -> Dict:
        """
        Make an async API call to Gemini for bias analysis
        """
        try:
            api_key = self.get_next_api_key()
            
            # Prepare the prompt
            articles_json = self.prepare_articles_for_prompt(articles_batch)
            user_prompt = self.get_user_prompt_template().format(articles_json=articles_json)
            
            # Gemini API request payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": self.get_system_prompt()},
                            {"text": user_prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,  # Low temperature for consistent JSON output
                    "topP": 0.8,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json"  # Force JSON response
                }
            }
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Extract the generated content
                    if 'candidates' in result and len(result['candidates']) > 0:
                        content = result['candidates'][0]['content']['parts'][0]['text']
                        
                        try:
                            
                            bias_results = json.loads(content)
                            
                            # Validate the response structure
                            if isinstance(bias_results, list) and len(bias_results) == len(articles_batch):
                                return {
                                    'success': True,
                                    'batch_id': batch_id,
                                    'results': bias_results,
                                    'api_key_used': api_key[-10:]  # Last 10 chars for logging
                                }
                            else:
                                logging.error(f"Batch {batch_id}: Invalid response structure")
                                return {'success': False, 'batch_id': batch_id, 'error': 'Invalid response structure'}
                        
                        except json.JSONDecodeError as e:
                            logging.error(f"Batch {batch_id}: JSON decode error: {e}")
                            return {'success': False, 'batch_id': batch_id, 'error': f'JSON decode error: {e}'}
                    
                    else:
                        logging.error(f"Batch {batch_id}: No candidates in response")
                        return {'success': False, 'batch_id': batch_id, 'error': 'No candidates in response'}
                
                else:
                    error_text = await response.text()
                    logging.error(f"Batch {batch_id}: API error {response.status}: {error_text}")
                    return {'success': False, 'batch_id': batch_id, 'error': f'API error {response.status}'}
        
        except Exception as e:
            logging.error(f"Batch {batch_id}: Exception occurred: {e}")
            return {'success': False, 'batch_id': batch_id, 'error': str(e)}

    def create_batches(self, df: pd.DataFrame, batch_size: int = 10) -> List[List[Dict]]:
        """
        Create batches of articles for processing
        """
        articles = df.to_dict('records')
        batches = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batches.append(batch)
        
        logging.info(f"Created {len(batches)} batches of {batch_size} articles each")
        return batches

    async def process_batches(self, batches: List[List[Dict]], max_concurrent: int = 4) -> List[Dict]:
        """
        Process batches with controlled concurrency and rate limiting
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_batch(session, batch, batch_id):
            async with semaphore:
                return await self.call_gemini_api(session, batch, batch_id)
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=100)
        ) as session:
            
            # Process batches in chunks to respect rate limits
            chunk_size = 50  # 50 requests per minute
            
            for chunk_start in range(0, len(batches), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(batches))
                chunk_batches = batches[chunk_start:chunk_end]
                
                logging.info(f"Processing chunk {chunk_start//chunk_size + 1}: batches {chunk_start} to {chunk_end-1}")
                
                # Create tasks for this chunk
                tasks = [
                    process_single_batch(session, batch, chunk_start + i)
                    for i, batch in enumerate(chunk_batches)
                ]
                
                # Execute chunk
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in chunk_results:
                    if isinstance(result, Exception):
                        logging.error(f"Task failed with exception: {result}")
                        results.append({'success': False, 'error': str(result)})
                    else:
                        results.append(result)
                
                # Log progress
                successful = sum(1 for r in chunk_results if isinstance(r, dict) and r.get('success', False))
                logging.info(f"Chunk completed: {successful}/{len(chunk_results)} successful")
                
                # Wait before next chunk (rate limiting)
                if chunk_end < len(batches):
                    logging.info("Waiting 60 seconds before next chunk...")
                    await asyncio.sleep(60)
        
        return results

    def save_results(self, original_df: pd.DataFrame, results: List[Dict], output_file: str):
        """
        Save the bias annotation results back to the DataFrame
        """
        # Create new columns for bias scores
        original_df['bias_left'] = 0.0
        original_df['bias_center'] = 0.0
        original_df['bias_right'] = 0.0
        original_df['bias_explanation'] = ''
        original_df['processing_status'] = 'failed'
        
        article_index = 0
        
        for batch_result in results:
            if batch_result.get('success', False):
                batch_annotations = batch_result['results']
                
                for annotation in batch_annotations:
                    if article_index < len(original_df):
                        # Update the DataFrame with bias scores
                        original_df.loc[article_index, 'bias_left'] = annotation['bias_scores']['left']
                        original_df.loc[article_index, 'bias_center'] = annotation['bias_scores']['center']
                        original_df.loc[article_index, 'bias_right'] = annotation['bias_scores']['right']
                        original_df.loc[article_index, 'bias_explanation'] = annotation['explanation']
                        original_df.loc[article_index, 'processing_status'] = 'success'
                        article_index += 1
            else:
                # Mark batch articles as failed
                batch_size = 10
                for _ in range(batch_size):
                    if article_index < len(original_df):
                        original_df.loc[article_index, 'processing_status'] = f"failed: {batch_result.get('error', 'unknown error')}"
                        article_index += 1
        
        # Save to CSV
        original_df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        
        # Print summary statistics
        success_count = len(original_df[original_df['processing_status'] == 'success'])
        total_count = len(original_df)
        
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total articles: {total_count}")
        print(f"Successfully processed: {success_count}")
        print(f"Success rate: {success_count/total_count*100:.2f}%")
        print(f"Results saved to: {output_file}")

async def main():
    """
    Main function to orchestrate the bias annotation process
    """
    
    # CONFIGURATION - UPDATE THESE VALUES
    CSV_FILE = "indian_news_articles.csv"  # Your input CSV file
    OUTPUT_FILE = "annotated_news_articles.csv"  # Output file with bias scores
    
    # Your 4 Gemini API keys
    API_KEYS = [
        "YOUR_API_KEY_1",
        "YOUR_API_KEY_2", 
        "YOUR_API_KEY_3",
        "YOUR_API_KEY_4"
    ]
        
    try:
        # Load the CSV file
        logging.info(f"Loading CSV file: {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)
        
        # Validate required columns
        required_columns = ['url', 'source', 'publish_date', 'title', 'text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return
        
        logging.info(f"Loaded {len(df)} articles")
        
        # Initialize the annotator
        annotator = GeminiBiasAnnotator(API_KEYS)
        
        # Create batches
        batches = annotator.create_batches(df, batch_size=10)
        
        # Estimate processing time
        estimated_minutes = len(batches) / 50  # 50 batches per minute
        logging.info(f"Estimated processing time: {estimated_minutes:.1f} minutes")
        
        # Process all batches
        logging.info("Starting bias annotation process...")
        start_time = datetime.now()
        
        results = await annotator.process_batches(batches)
        
        end_time = datetime.now()
        processing_time = end_time - start_time
        logging.info(f"Processing completed in {processing_time}")
        
        # Save results
        annotator.save_results(df, results, OUTPUT_FILE)
        
        # API usage summary
        print(f"\n=== API USAGE SUMMARY ===")
        for i, (key, requests) in enumerate(zip(API_KEYS, annotator.requests_per_key)):
            print(f"API Key {i+1} (...{key[-10:]}): {requests} requests")
        
    except FileNotFoundError:
        logging.error(f"CSV file not found: {CSV_FILE}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Example usage for testing with a small dataset
def test_with_sample_data():
    """
    Test function with sample data - useful for testing before processing the full dataset
    """
    # Create sample data
    sample_data = {
        'url': ['http://example1.com', 'http://example2.com', 'http://example3.com'],
        'source': ['Times of India', 'The Hindu', 'Indian Express'],
        'publish_date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'title': [
            'Government Announces New Economic Policy',
            'Opposition Criticizes Healthcare Budget',
            'Cricket Match Results from Mumbai'
        ],
        'text': [
            'The government today announced a comprehensive economic policy aimed at boosting growth...',
            'Opposition leaders heavily criticized the allocated healthcare budget saying it was insufficient...',
            'In a thrilling cricket match at Mumbai stadium, the home team secured victory...'
        ]
    }
    
    # Save sample data
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_news.csv', index=False)
    
    print("Sample data created in 'sample_news.csv'")
    print("Update the CSV_FILE variable in main() to 'sample_news.csv' to test")

if __name__ == "__main__":
    print("Political Bias Annotation Tool for Indian News Articles")
    print("=" * 60)
    
    # Uncomment the line below to create sample data for testing
    # test_with_sample_data()
    
    # Run the main annotation process
    asyncio.run(main())

# PROMPT TEMPLATES FOR REFERENCE
# These are the exact prompts used by the script above

SYSTEM_PROMPT = """You are a political bias analysis expert specializing in Indian news media. Your task is to analyze news articles and determine their political bias on a left-center-right spectrum specific to the Indian political context.

CRITICAL INSTRUCTIONS:
1. You MUST respond with ONLY valid JSON - no explanations, no markdown, no extra text
2. Analyze each article for political bias in the Indian context (BJP/right vs Congress/left vs neutral)
3. Assign bias scores that sum to approximately 1.0
4. Be conservative - if genuinely neutral, assign high center score
5. Consider: word choice, framing, source selection, emphasis, and tone
6. Your response must be a JSON array with exactly the same number of objects as input articles"""

USER_PROMPT_TEMPLATE = """Analyze these Indian news articles for political bias. Return ONLY a JSON array with this exact structure:

[
  {
    "article_id": 0,
    "bias_scores": {
      "left": 0.0,
      "center": 0.0,
      "right": 0.0
    },
    "explanation": "Brief explanation in 1-2 sentences"
  }
]

Articles to analyze:
{articles_json}

Remember: 
- JSON ONLY response
- bias_scores must sum to ~1.0
- Use Indian political context (BJP=right, Congress=left)
- Be conservative with bias detection"""