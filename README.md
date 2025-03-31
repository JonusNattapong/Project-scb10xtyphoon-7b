# SCB10X Typhoon-7B Model Fine-tuning

โครงการปรับแต่งโมเดล Typhoon-7B ด้วยเทคนิคขั้นสูงจากงานวิจัยล่าสุด เพื่อให้มีความสามารถในการเข้าใจและตอบคำถามภาษาไทยได้ดียิ่งขึ้น

## Workflow Overview

```mermaid
graph TD
    A[Input Data] --> B[prepare_dataset.bat]
    B --> C{เลือกประเภทข้อมูล}
    C --> D[Text Dataset]
    C --> E[Conversation Dataset]
    C --> F[Instruction Dataset]
    C --> G[Vision Dataset]
    C --> H[Teacher-Student Dataset]
    
    subgraph "Data Processing"
        D --> I[Clean Text]
        E --> I
        F --> I
        G --> J[Process Images]
        H --> K[Generate Teacher Outputs]
        
        J --> I
        K --> I
        
        I --> L[Tokenization]
        L --> M[Train/Test Split]
    end
    
    M --> N[Processed Dataset]
    N --> O[Training]
```

## Text Dataset Workflow

```mermaid
graph LR
    A[Input .txt] --> B[Clean Text]
    B --> C[Remove Extra Whitespace]
    C --> D[Handle Newlines]
    D --> E[Tokenization]
    E --> F[Train/Test Split]
```

## Conversation Dataset Workflow

```mermaid
graph LR
    A[Input .csv] --> B[Extract QA Pairs]
    B --> C[Format Template]
    C --> D[Clean Text]
    D --> E[Optional: Add Images]
    E --> F[Tokenization]
    F --> G[Train/Test Split]
```

## Vision Dataset Workflow

```mermaid
graph TD
    A[Input .csv] --> B[Load Images]
    B --> C[Process Images]
    C --> D[Resize if needed]
    D --> E[Convert to RGB]
    E --> F[Base64 Encoding]
    
    A --> G[Extract Text/Captions]
    G --> H[Clean Text]
    
    F --> I[Combine Data]
    H --> I
    I --> J[Tokenization]
    J --> K[Train/Test Split]
```

## Teacher-Student Learning Workflow

```mermaid
graph TD
    A[Input Text] --> B[Load Teacher Model]
    B --> C[Generate Logits]
    C --> D[Apply Softmax]
    
    A --> E[Clean Text]
    E --> F[Student Training Data]
    
    D --> G[Knowledge Distillation]
    F --> G
    G --> H[Final Dataset]
```

## คุณสมบัติหลัก

- ใช้เทคนิคจากโมเดลล่าสุด: LLaMA-3, Claude 3.5, Gemma, DeepSeek, Phi-3
- รองรับการปรับแต่งด้วย PEFT (LoRA, Prefix Tuning)
- ใช้เทคนิค QLoRA เพื่อลดการใช้หน่วยความจำ
- การปรับแต่งด้วย RLHF (Reinforcement Learning from Human Feedback)
- รองรับการเทรนแบบหลายขั้นตอน (Multi-stage Training)
- การเพิ่มประสิทธิภาพการอนุมานด้วย Flash Attention
- ชุดข้อมูลภาษาไทยจากหลายแหล่ง

## การติดตั้ง

```bash
# Clone โครงการ
git clone https://github.com/yourusername/Project-scb10xtyphoon-7b.git
cd Project-scb10xtyphoon-7b

# ติดตั้ง dependencies
pip install -r requirements.txt
```

## การใช้งาน

### การเทรนโมเดลพื้นฐาน

```bash
# สำหรับ Linux/Mac
./run_training.sh

# สำหรับ Windows
run_training.bat
```

### การเทรนโมเดลขั้นสูง

```bash
# สำหรับ Windows
run_advanced_training.bat --use_flash_attn --use_llama3

# สำหรับ Linux/Mac
python advanced_training.py --output_dir ./fine_tuned_typhoon_advanced --use_flash_attn --use_llama3
```

### การใช้งานโมเดลที่ปรับแต่งแล้ว

```bash
# โหมดสนทนาโต้ตอบ
python typhoon_inference.py --model_path ./fine_tuned_typhoon --interactive

# การประมวลผลไฟล์คำถาม
python typhoon_inference.py --model_path ./fine_tuned_typhoon --input_file questions.txt --output_file answers.txt
```

## เทคนิคที่ใช้ในโครงการ

- **DeepSeek Techniques**: LoRA ที่ปรับแต่งรูปแบบพิเศษ และการจัดการหน่วยความจำขั้นสูง
- **Meta AI**: Prefix Tuning แบบพิเศษจาก LLaMA-3
- **Anthropic**: Constitutional AI และเทคนิคจาก Claude 3.5
- **Microsoft**: Grouped-Query Attention (GQA) จาก Phi-3
- **Google**: Gated Mixture of Experts จาก Gemma
- **RWKV**: Linear Attention แบบไม่ใช้ attention matrix
- **EleutherAI**: Loss Adaptation เพื่อการเทรนที่มีประสิทธิภาพ
- **Mistral AI**: Sliding Window Attention เพื่อประมวลผลข้อความยาวได้อย่างมีประสิทธิภาพ

## ประเภทไฟล์

- `typhoon_model_training.py`: ไฟล์หลักสำหรับการเทรนโมเดล
- `typhoon_inference.py`: สำหรับการอนุมานและใช้งานโมเดล
- `data_preprocessing.py`: เตรียมและจัดการชุดข้อมูล
- `deepseek_utils.py`: เครื่องมือและฟังก์ชันจาก DeepSeek และงานวิจัยอื่นๆ
- `advanced_training.py`: การเทรนขั้นสูงที่รวมเทคนิคหลายอย่าง
- `run_training.sh` / `run_training.bat`: สคริปต์เริ่มการเทรนบน Linux/Windows
- `run_advanced_training.bat`: สคริปต์เริ่มการเทรนขั้นสูงบน Windows
- `run_inference.bat`: สคริปต์สำหรับการอนุมานบน Windows

## ข้อมูลอ้างอิง

- [SCB10X Typhoon-7B](https://huggingface.co/scb10x/typhoon-7b)
- [DeepSeek Research](https://www.deepseek.com/)
- [LLaMA-3 Technical Report](https://ai.meta.com/blog/meta-llama-3/)
- [Anthropic Claude 3.5 Technical Report](https://www.anthropic.com/claude)
- [EleutherAI Research](https://www.eleuther.ai/)
- [Microsoft Phi-3 Technical Report](https://www.microsoft.com/en-us/research/blog/phi-3-technical-report/)
