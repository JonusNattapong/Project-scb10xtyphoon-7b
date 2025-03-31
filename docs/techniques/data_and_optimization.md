# Data & Optimization Techniques

เอกสารนี้อธิบายเทคนิคที่เกี่ยวข้องกับการจัดการข้อมูลและการเพิ่มประสิทธิภาพที่ใช้ในโปรเจกต์นี้

## 1. Knowledge Distillation (Teacher-Student Learning)

-   **ที่มา:** แนวคิดพื้นฐานจาก Paper "Distilling the Knowledge in a Neural Network" และการประยุกต์ใช้กับ LLMs
-   **โมเดลครูที่ใช้:** `microsoft/OmniParser-v2.0` (สามารถเปลี่ยนได้)
-   **คำอธิบาย:** เทคนิคการถ่ายทอดความรู้จากโมเดลขนาดใหญ่ (Teacher) ไปยังโมเดลขนาดเล็ก (Student) โดยให้ Student เรียนรู้ที่จะทำนาย output distribution (logits/probabilities) ของ Teacher แทนที่จะเรียนรู้จาก ground truth labels โดยตรง ช่วยให้ Student มีประสิทธิภาพใกล้เคียง Teacher แต่มีขนาดเล็กกว่า
-   **การนำไปใช้:**
    -   `data_preprocessing.py` -> `generate_teacher_outputs()`: ใช้ Teacher model สร้าง logits สำหรับข้อมูล input
    -   `data_preprocessing.py` -> `prepare_teacher_student_dataset()`: เตรียม dataset ที่มีทั้ง input text และ teacher logits
    -   (ส่วนการเทรน Student model ด้วย distillation loss ยังไม่ได้ implement ใน `advanced_training.py` หรือ `vision_training.py` ปัจจุบัน)
-   **วิธีเปิดใช้งาน:**
    -   เตรียมข้อมูล: `prepare_dataset.bat` -> เลือก "Teacher-Student"
    -   การเทรน: ต้องปรับปรุง training script ให้ใช้ distillation loss (เช่น KL Divergence)

## 2. Constitutional AI Filtering (Anthropic)

-   **ที่มา:** Anthropic's Constitutional AI principles
-   **คำอธิบาย:** ใช้ชุดของกฎหรือหลักการ (Constitution) เพื่อกรองข้อมูลที่ไม่เหมาะสมหรือเป็นอันตรายออกจากชุดข้อมูลการเทรน ช่วยให้โมเดลที่เทรนมีความปลอดภัยและสอดคล้องกับหลักจริยธรรมมากขึ้น
-   **การนำไปใช้:**
    -   `deepseek_utils.py` -> `apply_anthropic_constitutional_ai_filter()`: ฟังก์ชันสำหรับกรอง dataset (ปัจจุบันใช้ logic placeholder)
    -   `advanced_training.py` -> `prepare_advanced_datasets()`: เรียกใช้ฟังก์ชันกรอง
    -   `vision_training.py` -> `_constitutional_ai_check()`: Placeholder สำหรับตรวจสอบ batch ระหว่างเทรน
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ `prepare_advanced_datasets()` โดยอัตโนมัติ (แต่ logic การกรองจริงยังต้อง implement เพิ่มเติม)

## 3. Flash Attention (Meta AI)

-   **ที่มา:** Paper "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
-   **คำอธิบาย:** การ implement attention mechanism ที่มีประสิทธิภาพสูง ลดการใช้หน่วยความจำและเพิ่มความเร็วในการคำนวณ โดยเฉพาะบน GPU รุ่นใหม่ๆ
-   **การนำไปใช้:**
    -   `deepseek_utils.py` -> `load_model_with_advanced_techniques()`: เปิดใช้งานผ่าน `attn_implementation="flash_attention_2"` (สำหรับ Text)
    -   `vision_training.py` -> `VisionTrainer.__init__()`: เปิดใช้งานผ่าน `to_bettertransformer()` (สำหรับ Vision)
-   **วิธีเปิดใช้งาน:**
    -   Text: `--use_flash_attn` ใน `run_advanced_training.bat`
    -   Vision: `--use_flash_attention` ใน `train_vision_model.bat` (เปิดใช้งานโดย default ถ้า hardware รองรับ)

## 4. 8-bit Adam Optimizer (Meta AI)

-   **ที่มา:** `bitsandbytes` library
-   **คำอธิบาย:** ใช้ Adam optimizer เวอร์ชั่นที่เก็บ state ในรูปแบบ 8-bit แทน 32-bit ช่วยลดการใช้หน่วยความจำของ optimizer ลงอย่างมาก (ประมาณ 4 เท่า)
-   **การนำไปใช้:**
    -   `vision_training.py` -> `VisionTrainer.train()`: เลือก optimizer class
    -   (สำหรับ Text `Trainer` ของ Hugging Face อาจรองรับผ่าน `TrainingArguments`)
-   **วิธีเปิดใช้งาน:**
    -   Vision: `--use_8bit_adam` ใน `train_vision_model.bat` (เปิดใช้งานโดย default ถ้าติดตั้ง `bitsandbytes` สำเร็จ)
    -   Text: อาจต้องตั้งค่าใน `TrainingArguments` (ตรวจสอบเอกสาร `transformers`)

## 5. Safetensors

-   **ที่มา:** Hugging Face
-   **คำอธิบาย:** รูปแบบไฟล์ใหม่สำหรับการบันทึก model weights ที่ปลอดภัยกว่า (ป้องกัน arbitrary code execution จาก pickle) เร็วกว่า และมีขนาดเล็กกว่า `.bin` หรือ `.pt` แบบเดิม
-   **การนำไปใช้:**
    -   `vision_utils.py` -> `save_model()`, `load_model()`: ใช้ `safetensors.torch.save_file` และ `load_file`
    -   `vision_training.py` -> `_save_checkpoint()`: ใช้ `pipeline.save_pretrained(safe_serialization=True)`
    -   (สำหรับ Text `Trainer` ของ Hugging Face อาจรองรับผ่าน `TrainingArguments`)
-   **วิธีเปิดใช้งาน:** เป็น default สำหรับ Vision training checkpoints และ `vision_utils.save_model()`

## 6. Distributed Training (Accelerate)

-   **ที่มา:** Hugging Face Accelerate Library
-   **คำอธิบาย:** ช่วยให้การเทรนโมเดลบนหลาย GPU หรือหลายเครื่อง (multi-node) ทำได้ง่ายขึ้น จัดการการกระจายข้อมูลและโมเดลโดยอัตโนมัติ
-   **การนำไปใช้:**
    -   `vision_training.py`: ใช้ `Accelerator` object จัดการ model, optimizer, dataloader, logging, saving
    -   `advanced_training.py`: `Trainer` ของ Hugging Face ใช้ `accelerate` เบื้องหลัง
    -   สคริปต์ `.bat`: ใช้ `accelerate launch` แทน `python`
-   **วิธีเปิดใช้งาน:** รันผ่าน `accelerate launch` (ทำโดยอัตโนมัติใน `.bat` script) และ config ผ่าน `accelerate config`
