@echo off
setlocal enabledelayedexpansion

echo ================================
echo Data Preparation for Training
echo ================================

echo Step 1: Download datasets from Hugging Face
python download_datasets.py

echo.
echo Step 2: Process downloaded datasets
echo.
echo Processing text datasets...
python data_preprocessing.py --input_file datasets/text/thai_wikipedia --input_type text --output_dir outputs/processed_data --dataset_type text
python data_preprocessing.py --input_file datasets/text/thaigpt4 --input_type text --output_dir outputs/processed_data --dataset_type text

echo.
echo Processing conversation datasets...
python data_preprocessing.py --input_file datasets/conversation/belle_thai --input_type csv --output_dir outputs/processed_data --dataset_type conversation
python data_preprocessing.py --input_file datasets/conversation/thai_alpaca --input_type csv --output_dir outputs/processed_data --dataset_type conversation

echo.
echo Processing vision datasets...
python data_preprocessing.py --input_file datasets/vision/laion_thai --input_type csv --output_dir outputs/processed_data --dataset_type vision --image_column image --text_column text --caption_column caption
python data_preprocessing.py --input_file datasets/vision/thai_art --input_type csv --output_dir outputs/processed_data --dataset_type vision --image_column image --text_column text --caption_column caption

echo.
echo Processing instruction datasets...
python data_preprocessing.py --input_file datasets/instruction/thaiinstruct --input_type json --output_dir outputs/processed_data --dataset_type instruction
python data_preprocessing.py --input_file datasets/instruction/thai_dolly --input_type json --output_dir outputs/processed_data --dataset_type instruction

echo.
echo Data preparation completed!
echo Check outputs/processed_data/ directory for the processed datasets.
echo Check outputs/logs/data_preprocessing/ directory for processing details.

pause
