set -euo pipefail
export PYTHONUNBUFFERED=1


# Load environment variables (.env)
if [ -f .env ]; then
  echo "🔧 Loading environment variables from .env..."
  export $(grep -v '^#' .env | xargs)

#RUnning the flow 


python src/MultiThreaded_News_Scrapper.py --in data/raw --out data/interim
echo "Scraped articles saved to data/interim/"

python src/Article_annotator.py --in data/interim --out data/processed/annotated.csv
echo "✅ Labeled dataset saved to data/processed/annotated.csv"

python src/train.py  --data data/processed/annotated.csv --out models/final
echo "✅ Model saved to models"

echo "🎉 Pipeline finished successfully!"
