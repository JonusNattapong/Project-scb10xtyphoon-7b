import os
from datasets import load_dataset
import logging
from tqdm import tqdm
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASETS = {
    "text": [
        {
            "name": "Thai Wikipedia",
            "id": "wikipedia",
            "config": "20230601.th",
            "split": "train"
        },
        {
            "name": "ThaiGPT4",
            "id": "pythainlp/thaigpt4",
            "split": "train"
        }
    ],
    "conversation": [
        {
            "name": "BELLE Thai",
            "id": "KoichiYasuoka/belle-thai",
            "split": "train"
        },
        {
            "name": "Thai Alpaca",
            "id": "Mookor/thai-alpaca-lora",
            "split": "train"
        }
    ],
    "vision": [
        {
            "name": "LAION Thai",
            "id": "laion/laion2B-multi",
            "split": "train",
            "filter": {"language": "th"}
        },
        {
            "name": "Thai Art",
            "id": "NattKhem/thai-art",
            "split": "train"
        }
    ],
    "instruction": [
        {
            "name": "ThaiInstruct",
            "id": "wangchangshot/thai-instruct",
            "split": "train"
        },
        {
            "name": "Thai Dolly",
            "id": "wangchangshot/thai-dolly",
            "split": "train"
        }
    ]
}

def download_dataset(dataset_info, output_dir):
    """ดาวน์โหลดและบันทึกชุดข้อมูล"""
    logger.info(f"Downloading {dataset_info['name']}...")
    
    try:
        # โหลดจาก Hugging Face
        if "config" in dataset_info:
            dataset = load_dataset(
                dataset_info["id"],
                dataset_info["config"],
                split=dataset_info["split"]
            )
        else:
            dataset = load_dataset(
                dataset_info["id"],
                split=dataset_info["split"]
            )
        
        # กรองข้อมูลถ้ามีการระบุ filter
        if "filter" in dataset_info:
            dataset = dataset.filter(
                lambda x: all(x[k] == v for k, v in dataset_info["filter"].items())
            )
        
        # บันทึกข้อมูล
        save_path = os.path.join(output_dir, dataset_info["name"].lower().replace(" ", "_"))
        dataset.save_to_disk(save_path)
        
        # บันทึก metadata
        with open(f"{save_path}/metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "name": dataset_info["name"],
                "id": dataset_info["id"],
                "split": dataset_info["split"],
                "size": len(dataset),
                "columns": dataset.column_names
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(dataset)} examples to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {dataset_info['name']}: {str(e)}")
        return False

def main():
    # สร้างโฟลเดอร์สำหรับแต่ละประเภทข้อมูล
    base_dir = "datasets"
    os.makedirs(base_dir, exist_ok=True)
    
    for data_type, datasets in DATASETS.items():
        logger.info(f"\nProcessing {data_type} datasets...")
        type_dir = os.path.join(base_dir, data_type)
        os.makedirs(type_dir, exist_ok=True)
        
        # ดาวน์โหลดแต่ละชุดข้อมูล
        results = []
        for dataset_info in datasets:
            success = download_dataset(dataset_info, type_dir)
            results.append({
                "name": dataset_info["name"],
                "success": success
            })
        
        # บันทึกสรุปผล
        with open(os.path.join(type_dir, "download_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "type": data_type,
                "total": len(datasets),
                "successful": sum(1 for r in results if r["success"]),
                "results": results
            }, f, ensure_ascii=False, indent=2)
    
    logger.info("\nDownload completed! Summary:")
    for data_type in DATASETS:
        summary_path = os.path.join(base_dir, data_type, "download_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                logger.info(f"{data_type}: {summary['successful']}/{summary['total']} datasets downloaded")

if __name__ == "__main__":
    main()
