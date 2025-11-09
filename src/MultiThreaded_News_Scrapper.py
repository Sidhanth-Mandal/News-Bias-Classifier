import pandas as pd
import numpy as np
from newspaper import Article, Config
import newspaper
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from queue import Queue

class NewsArticleScraper:
    def __init__(self, output_csv='scraped_articles.csv', failed_urls_file='failed_urls.txt', 
                 batch_size=50, workers_per_csv=5, max_retries=3):
        self.output_csv = output_csv
        self.failed_urls_file = failed_urls_file
        self.batch_size = batch_size
        self.workers_per_csv = workers_per_csv
        self.max_retries = max_retries
        
        # Thread-safe locks for file operations
        self.csv_lock = threading.Lock()
        self.failed_lock = threading.Lock()
        
        # Setup logging with thread safety
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize output CSV with headers if it doesn't exist
        if not os.path.exists(self.output_csv):
            with self.csv_lock:
                pd.DataFrame(columns=['url', 'source', 'publish_date', 'title', 'text']).to_csv(
                    self.output_csv, index=False
                )
        
        # Statistics tracking
        self.stats_lock = threading.Lock()
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0

    def create_session(self):
        """Create a session with retry strategy for each thread"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def scrape_single_article(self, row_data, session=None):
        """Scrape a single article with retry logic"""
        url, source, publish_date = row_data
        
        if session is None:
            session = self.create_session()
        
        for attempt in range(self.max_retries):
            try:
                # Random delay to be respectful to servers
                time.sleep(random.uniform(0.1, 0.3))
                
                # Create article object with custom config
                config = newspaper.Config()
                config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                config.request_timeout = 10
                
                article = Article(url, config=config)
                
                # Download and parse
                article.download()
                article.parse()
                
                # Extract data
                title = article.title.strip() if article.title else ""
                text = article.text.strip() if article.text else ""
                
                # Basic validation
                if not title and not text:
                    raise Exception("No content extracted")
                
                self.logger.info(f"Successfully scraped: {url[:60]}... (Source: {source})")
                
                # Update statistics
                with self.stats_lock:
                    self.total_processed += 1
                    self.total_successful += 1
                
                return {
                    'url': url,
                    'source': source,
                    'publish_date': publish_date,
                    'title': title,
                    'text': text,
                    'status': 'success'
                }
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(random.uniform(1, 3))  # Longer delay between retries
                else:
                    # Log failed URL
                    self.log_failed_url(url, str(e))
                    
                    # Update statistics
                    with self.stats_lock:
                        self.total_processed += 1
                        self.total_failed += 1
                    
                    return {
                        'url': url,
                        'source': source,
                        'publish_date': publish_date,
                        'title': '',
                        'text': '',
                        'status': 'failed'
                    }

    def log_failed_url(self, url, error):
        """Log failed URL to text file with thread safety"""
        with self.failed_lock:
            with open(self.failed_urls_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - {url} - Error: {error}\n")

    def save_batch(self, results, csv_filename):
        """Save a batch of results to CSV with thread safety"""
        if not results:
            return
            
        # Filter out failed results for CSV saving
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if successful_results:
            # Create clean results without status column
            clean_results = []
            for result in successful_results:
                clean_result = {
                    'url': result.get('url', ''),
                    'source': result.get('source', ''),
                    'publish_date': result.get('publish_date', ''),
                    'title': result.get('title', ''),
                    'text': result.get('text', '')
                }
                clean_results.append(clean_result)
                
            df_batch = pd.DataFrame(clean_results)
            
            # Thread-safe CSV writing
            with self.csv_lock:
                df_batch.to_csv(self.output_csv, mode='a', header=False, index=False, encoding='utf-8')
            
            self.logger.info(f"Saved batch of {len(successful_results)} articles from {csv_filename}")

    def process_single_csv_file(self, csv_file):
        """Process a single CSV file with dedicated workers"""
        try:
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_cols = ['url', 'source', 'publish_date']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {csv_file}: {missing_cols}")
            
            self.logger.info(f"Starting processing {len(df)} URLs from {csv_file} with {self.workers_per_csv} workers")
            
            # Prepare data for processing
            url_data = [(row['url'], row['source'], row['publish_date']) 
                       for _, row in df.iterrows()]
            
            results_batch = []
            processed_count = 0
            
            # Process URLs with ThreadPoolExecutor (dedicated workers for this CSV)
            with ThreadPoolExecutor(max_workers=self.workers_per_csv, 
                                  thread_name_prefix=f"CSV-{os.path.basename(csv_file)}") as executor:
                
                # Create session for each worker thread
                sessions = {i: self.create_session() for i in range(self.workers_per_csv)}
                session_queue = Queue()
                for session in sessions.values():
                    session_queue.put(session)
                
                def scrape_with_session(url_info):
                    session = session_queue.get()
                    try:
                        result = self.scrape_single_article(url_info, session)
                        return result
                    finally:
                        session_queue.put(session)
                
                # Submit all tasks
                future_to_url = {executor.submit(scrape_with_session, url_info): url_info 
                               for url_info in url_data}
                
                # Process completed tasks
                for future in as_completed(future_to_url):
                    try:
                        result = future.result()
                        results_batch.append(result)
                        processed_count += 1
                        
                        # Save in batches
                        if len(results_batch) >= self.batch_size:
                            self.save_batch(results_batch, csv_file)
                            results_batch = []
                            
                        # Progress update every 25 processed URLs
                        if processed_count % 25 == 0:
                            success_rate = (len([r for r in results_batch if r['status'] == 'success']) / 
                                          len(results_batch) * 100) if results_batch else 0
                            self.logger.info(f"{csv_file}: Processed {processed_count}/{len(df)} URLs")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing future for {csv_file}: {e}")
                        
            # Save remaining results
            if results_batch:
                self.save_batch(results_batch, csv_file)
                
            successful_from_this_csv = len([r for r in results_batch if r.get('status') == 'success'])
            self.logger.info(f"Completed {csv_file}: {processed_count} URLs processed, "
                           f"{successful_from_this_csv} successful")
            
        except Exception as e:
            self.logger.error(f"Error processing {csv_file}: {e}")

    def process_multiple_csv_files_parallel(self, csv_files):
        """Process multiple CSV files in parallel, each with its own worker pool"""
        self.logger.info(f"Starting parallel processing of {len(csv_files)} CSV files")
        self.logger.info(f"Total workers: {len(csv_files)} × {self.workers_per_csv} = {len(csv_files) * self.workers_per_csv}")
        
        start_time = datetime.now()
        
        # Use ThreadPoolExecutor to process CSV files in parallel
        with ThreadPoolExecutor(max_workers=len(csv_files), 
                              thread_name_prefix="CSVProcessor") as csv_executor:
            
            # Submit each CSV file for processing
            csv_futures = {csv_executor.submit(self.process_single_csv_file, csv_file): csv_file 
                          for csv_file in csv_files}
            
            # Wait for all CSV files to complete
            for future in as_completed(csv_futures):
                csv_file = csv_futures[future]
                try:
                    future.result()
                    self.logger.info(f"Completed processing: {csv_file}")
                except Exception as e:
                    self.logger.error(f"Error in CSV file {csv_file}: {e}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"All CSV files processing completed in {duration}")

    def get_summary_stats(self):
        """Get summary statistics of the scraping process"""
        try:
            # Count successful articles from output file
            successful_count = 0
            if os.path.exists(self.output_csv):
                df_output = pd.read_csv(self.output_csv)
                successful_count = len(df_output)
                
            # Count failed URLs
            failed_count = 0
            if os.path.exists(self.failed_urls_file):
                with open(self.failed_urls_file, 'r') as f:
                    failed_count = len(f.readlines())
                    
            return {
                'successful_articles': successful_count,
                'failed_urls': failed_count,
                'total_processed': successful_count + failed_count,
                'live_stats': {
                    'processed': self.total_processed,
                    'successful': self.total_successful,
                    'failed': self.total_failed
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting summary stats: {e}")
            return None

def main():
    """Main function to run the scraper"""
    
    # Configuration - Update with your 9 CSV files
    CSV_FILES = [
        'news_site1.csv',   # Replace with your actual CSV file names
        'news_site2.csv',
        'news_site3.csv',
        'news_site4.csv',
        'news_site5.csv',
        'news_site6.csv',
        'news_site7.csv',
        'news_site8.csv',
        'news_site9.csv'
    ]
    
    OUTPUT_CSV = 'scraped_articles.csv'
    FAILED_URLS_FILE = 'failed_urls.txt'
    BATCH_SIZE = 50
    WORKERS_PER_CSV = 5  # 5 workers per CSV file = 45 total workers
    
    # Initialize scraper
    scraper = NewsArticleScraper(
        output_csv=OUTPUT_CSV,
        failed_urls_file=FAILED_URLS_FILE,
        batch_size=BATCH_SIZE,
        workers_per_csv=WORKERS_PER_CSV
    )
    
    # Check if CSV files exist
    existing_files = [f for f in CSV_FILES if os.path.exists(f)]
    if not existing_files:
        print(f"Error: None of the specified CSV files exist.")
        print("Please update the CSV_FILES list with your actual file names:")
        for i, filename in enumerate(CSV_FILES, 1):
            print(f"  {i}. {filename}")
        return
    
    missing_files = [f for f in CSV_FILES if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: The following files don't exist and will be skipped:")
        for f in missing_files:
            print(f"  - {f}")
        print()
    
    print(f"Processing {len(existing_files)} CSV files with {WORKERS_PER_CSV} workers each")
    print(f"Total concurrent workers: {len(existing_files)} × {WORKERS_PER_CSV} = {len(existing_files) * WORKERS_PER_CSV}")
    print()
    
    # Process files
    start_time = datetime.now()
    print(f"Starting parallel scraper at {start_time}")
    
    try:
        scraper.process_multiple_csv_files_parallel(existing_files)
        
        # Print summary
        stats = scraper.get_summary_stats()
        if stats:
            print("\n" + "="*60)
            print("SCRAPING SUMMARY")
            print("="*60)
            print(f"Successful articles: {stats['successful_articles']}")
            print(f"Failed URLs: {stats['failed_urls']}")
            print(f"Total processed: {stats['total_processed']}")
            if stats['total_processed'] > 0:
                success_rate = stats['successful_articles'] / stats['total_processed'] * 100
                print(f"Success rate: {success_rate:.1f}%")
            
            print(f"\nLive processing stats:")
            print(f"  Processed: {stats['live_stats']['processed']}")
            print(f"  Successful: {stats['live_stats']['successful']}")
            print(f"  Failed: {stats['live_stats']['failed']}")
            
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nCompleted in {duration}")
        print(f"Results saved to: {OUTPUT_CSV}")
        print(f"Failed URLs logged to: {FAILED_URLS_FILE}")
        print(f"Detailed logs in: scraper.log")
        
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("SCRAPING INTERRUPTED BY USER")
        print("="*50)
        print("Progress has been saved incrementally.")
        print("You can restart the script to continue from where it left off.")
        
        # Show current stats
        stats = scraper.get_summary_stats()
        if stats:
            print(f"\nProgress so far:")
            print(f"  Successful articles: {stats['successful_articles']}")
            print(f"  Failed URLs: {stats['failed_urls']}")
            
    except Exception as e:
        print(f"Error during scraping: {e}")
        scraper.logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()