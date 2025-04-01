# User Guide

เอกสารนี้อธิบายวิธีใช้งานโปรเจกต์ SCB10X Typhoon-7B Fine-tuning สำหรับงานต่างๆ

## 1. การติดตั้ง (Setup)

ดูรายละเอียดการติดตั้ง Hardware และ Software ที่จำเป็นได้ที่ [Resource Estimation](./RESOURCE_ESTIMATION.md)

```bash
# 1. Clone Repository
git clone https://github.com/yourusername/Project-scb10xtyphoon-7b.git
cd Project-scb10xtyphoon-7b

# 2. สร้าง Virtual Environment (แนะนำ)
# Python venv
python -m venv env
source env/bin/activate  # Linux/Mac
# หรือ env\Scripts\activate # Windows

# Conda
# conda create -n typhoon python=3.9
# conda activate typhoon

# 3. ติดตั้ง Dependencies
pip install -r requirements.txt

# 4. (Optional) Configure Accelerate for Multi-GPU
accelerate config
```

## 2. การเตรียมข้อมูล (Data Preparation)

### 2.1 ดาวน์โหลดและเตรียมข้อมูลอัตโนมัติ (แนะนำ)

สคริปต์นี้จะดาวน์โหลดชุดข้อมูลภาษาไทยยอดนิยมจาก Hugging Face และประมวลผลให้พร้อมใช้งาน

```bash
# สำหรับ Windows
prepare_training_data.bat

# สำหรับ Linux/Mac (ต้องสร้าง .sh เอง หรือรันคำสั่งใน .bat ทีละบรรทัด)
# ตัวอย่าง:
# python download_datasets.py
# python data_preprocessing.py --hf_dataset wikipedia --config 20230601.th --output_dir outputs/processed_data --dataset_type text
# ... (รัน data_preprocessing.py สำหรับ dataset อื่นๆ) ...
```

-   ข้อมูลดิบจะถูกเก็บใน `datasets/`
-   ข้อมูลที่ประมวลผลแล้วจะอยู่ใน `outputs/processed_data/` (แยกตามประเภทและชื่อ dataset)
-   Log การทำงานอยู่ใน `outputs/logs/data_preprocessing/`

### 2.2 เตรียมข้อมูลด้วยตนเอง (Custom Data)

ใช้ `prepare_dataset.bat` (หรือ `data_preprocessing.py` โดยตรง)

```bash
# รันสคริปต์เตรียมข้อมูล (Windows)
prepare_dataset.bat
```

จากนั้นเลือกประเภทข้อมูลและทำตามขั้นตอน:

-   **Text:** ป้อน path ไฟล์ `.txt`
-   **Conversation:** ป้อน path ไฟล์ `.csv` (ต้องมีคอลัมน์ `question`, `answer`)
-   **Instruction:** ป้อน path ไฟล์ `.json` (ต้องมี key `instruction`, `output`, และ `input` (optional))
-   **Vision:** ป้อน path ไฟล์ `.csv` (ต้องมีคอลัมน์ `image` (path), `text` (optional), `caption` (optional)) และชื่อ Vision Processor (เช่น `openai/clip-vit-large-patch14`)
-   **Teacher-Student:** ป้อน path ไฟล์ `.txt` (จะใช้ `microsoft/OmniParser-v2.0` เป็น Teacher)

ผลลัพธ์จะถูกเก็บใน `outputs/processed_data/[dataset_type]/[dataset_name]/`

## 3. การเทรนโมเดล (Training)

### 3.1 เทรนโมเดล Text พื้นฐาน

ใช้สคริปต์ `run_training.bat` (หรือ `.sh`) ซึ่งจะใช้ `typhoon_model_training.py`

```bash
# สำหรับ Windows
run_training.bat

# สำหรับ Linux/Mac
./run_training.sh
```

-   ใช้การตั้งค่าพื้นฐานสำหรับการ fine-tune
-   ผลลัพธ์ (โมเดล) จะถูกเก็บใน `./fine_tuned_typhoon` (ตามค่า default ในสคริปต์)

### 3.2 เทรนโมเดล Text ขั้นสูง

ใช้สคริปต์ `run_advanced_training.bat` (หรือ `accelerate launch advanced_training.py`)

```bash
# ตัวอย่าง (Windows) - เปิด Flash Attention และเทคนิค LLaMA-3
run_advanced_training.bat --use_flash_attn --use_llama3

# ตัวอย่าง (Linux/Mac)
accelerate launch advanced_training.py --use_flash_attn --use_llama3 --output_dir outputs/models/text/my_advanced_run
```

-   ใช้ `accelerate` สำหรับ Multi-GPU training
-   สามารถระบุ arguments ต่างๆ เพื่อเปิดใช้งานเทคนิคเฉพาะ (ดู `--help`)
-   ผลลัพธ์ (โมเดล) จะถูกเก็บใน `outputs/models/text/[experiment_name]/`
-   Log การทำงานอยู่ใน `outputs/logs/text_training/`

### 3.3 เทรนโมเดล Vision ขั้นสูง

ใช้สคริปต์ `train_vision_model.bat` (หรือ `accelerate launch vision_training.py`)

```bash
# สำหรับ Windows
train_vision_model.bat
```

-   สคริปต์จะถามให้เลือก configuration (Basic, Advanced, Custom)
-   ใช้ `accelerate` สำหรับ Multi-GPU training
-   ผลลัพธ์ (checkpoints) จะถูกเก็บใน `outputs/models/vision/checkpoints/`
-   Log การทำงานอยู่ใน `outputs/logs/vision_training/`

## 4. การใช้งานโมเดล (Inference)

### 4.1 Inference โมเดล Text

ใช้สคริปต์ `run_inference.bat` (หรือ `python typhoon_inference.py`)

```bash
# โหมดโต้ตอบ (Windows)
run_inference.bat --model_path outputs/models/text/typhoon_advanced_YYYYMMDD_HHMMSS --interactive

# ประมวลผลไฟล์ (Windows)
run_inference.bat --model_path outputs/models/text/typhoon_advanced_YYYYMMDD_HHMMSS --input_file questions.txt --output_file answers.txt

# ตัวอย่าง (Linux/Mac)
python typhoon_inference.py --model_path outputs/models/text/typhoon_advanced_YYYYMMDD_HHMMSS --interactive
```

-   ระบุ `--model_path` ไปยังโมเดลที่เทรนเสร็จแล้ว
-   ใช้ `--interactive` สำหรับโหมดแชท
-   ใช้ `--input_file` และ `--output_file` สำหรับประมวลผลแบบ batch

### 4.2 Inference โมเดล Vision (Image Generation)

ใช้ `vision_utils.py` ผ่าน Python script

```python
from vision_utils import VisionEnhancer
from PIL import Image

# โหลดโมเดล (อาจต้องโหลด checkpoint ที่ดีที่สุด)
enhancer = VisionEnhancer()
# enhancer.load_model("outputs/models/vision/checkpoints/vision_model_epoch_X") # โหลด checkpoint

# 1. Text-to-Image
prompt = "วัดไทยสวยงามท่ามกลางทุ่งนาเขียวขจี"
image = enhancer.text_to_image(prompt, style_preset="photographic")
image.save("outputs/generated_images/temple_rice_field.jpg")

# 2. Image-and-Text-to-Image
source_img = Image.open("input_images/cat.jpg")
prompt = "A cat wearing a tiny traditional Thai hat, cute, detailed"
edited_image = enhancer.image_and_text_to_image(source_img, prompt, strength=0.7)
edited_image.save("outputs/generated_images/cat_with_hat.jpg")

# 3. ControlNet
control_img = Image.open("input_images/sketch.png") # ภาพโครงร่าง
prompt = "Detailed painting of a Thai mythical creature based on the sketch"
controlled_image = enhancer.control_image(prompt, control_img)
controlled_image.save("outputs/generated_images/mythical_creature.jpg")

# 4. Style Transfer
content_img = Image.open("input_images/portrait.jpg")
style = "Traditional Thai mural painting style"
styled_image = enhancer.style_transfer(content_img, style, strength=0.8)
styled_image.save("outputs/generated_images/portrait_thai_style.jpg")
```

-   สร้าง instance ของ `VisionEnhancer`
-   เรียกใช้ method ที่ต้องการ (`text_to_image`, `image_and_text_to_image`, `control_image`, `style_transfer`)
-   บันทึกภาพผลลัพธ์

## 5. การดูเอกสารอื่นๆ

-   **[API Documentation](./API.md):** รายละเอียดฟังก์ชันและคลาส
-   **[Deployment Guide](./DEPLOYMENT.md):** การนำไปใช้งานจริง
-   **[Testing Guide](./TESTING.md):** วิธีการทดสอบระบบ
-   **[Contributing Guidelines](../CONTRIBUTING.md):** แนวทางการร่วมพัฒนา
-   **[Techniques](./techniques/):** คำอธิบายเทคนิคขั้นสูงที่ใช้
