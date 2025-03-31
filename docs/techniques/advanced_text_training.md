# Advanced Text Training Techniques

เอกสารนี้อธิบายเทคนิคขั้นสูงที่ใช้ในการเทรนโมเดล Text (Typhoon-7B) ในโปรเจกต์นี้

## 1. PEFT (Parameter-Efficient Fine-Tuning)

-   **ที่มา:** Hugging Face PEFT Library
-   **คำอธิบาย:** เทคนิคการ fine-tune โมเดลขนาดใหญ่โดยปรับปรุงพารามิเตอร์เพียงส่วนน้อย ช่วยประหยัดทรัพยากร (หน่วยความจำ, เวลา) อย่างมากเมื่อเทียบกับการ fine-tune ทั้งโมเดล
-   **รูปแบบที่ใช้:**
    -   **LoRA (Low-Rank Adaptation):** เพิ่ม trainable low-rank matrices เข้าไปใน layers ของ Transformer ทำให้ปรับปรุงเฉพาะส่วนนี้แทนพารามิเตอร์เดิมทั้งหมด
    -   **Prefix Tuning:** เพิ่ม trainable prefix vectors เข้าไปใน input ของแต่ละ layer
-   **การนำไปใช้:** `advanced_training.py` -> `apply_advanced_peft_techniques()`
-   **วิธีเปิดใช้งาน:** `--peft_method lora` (default), `--peft_method prefix`, `--peft_method both`

## 2. QLoRA (Quantized LoRA)

-   **ที่มา:** Paper "QLoRA: Efficient Finetuning of Quantized LLMs"
-   **คำอธิบาย:** เป็นการผสมผสาน LoRA กับการ Quantization (ลดความแม่นยำของพารามิเตอร์ เช่น จาก 32-bit เป็น 4-bit หรือ 8-bit) ทำให้ลดการใช้หน่วยความจำลงได้อีกมาก โดยยังคงประสิทธิภาพใกล้เคียงเดิม
-   **การนำไปใช้:** `deepseek_utils.py` -> `load_model_with_advanced_techniques()` (ใช้ `bitsandbytes` library)
-   **วิธีเปิดใช้งาน:** `--quantization 4bit` (default), `--quantization 8bit`

## 3. Multi-stage Training (Inspired by TogetherAI)

-   **ที่มา:** แนวคิดจาก TogetherAI และงานวิจัยอื่นๆ
-   **คำอธิบาย:** แบ่งการเทรนออกเป็นหลายขั้นตอน (stages) เช่น Pre-training, Supervised Fine-Tuning (SFT), และ Reinforcement Learning (DPO) โดยแต่ละขั้นตอนอาจใช้ dataset, learning rate, หรือเทคนิคที่แตกต่างกัน เพื่อให้โมเดลเรียนรู้ความสามารถที่หลากหลาย
-   **การนำไปใช้:** `advanced_training.py` -> `train_with_advanced_techniques()`, `deepseek_utils.py` -> `create_together_ai_training_stages()`
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ workflow หลักใน `advanced_training.py`

## 4. DPO (Direct Preference Optimization) for RLHF

-   **ที่มา:** Paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
-   **คำอธิบาย:** เทคนิค Reinforcement Learning from Human Feedback (RLHF) รูปแบบใหม่ที่ไม่ต้องเทรน Reward Model แยกต่างหาก แต่ใช้การเปรียบเทียบข้อมูลคู่ (chosen/rejected) เพื่อปรับปรุงโมเดลโดยตรง ทำให้กระบวนการ RLHF ง่ายและเสถียรขึ้น
-   **การนำไปใช้:** `advanced_training.py` -> `train_with_advanced_techniques()` (ใช้ `DPOTrainer` จาก `trl` library)
-   **วิธีเปิดใช้งาน:** เป็น stage หนึ่งใน Multi-stage Training (ถ้ามีข้อมูล preference)

## 5. RWKV Linear Attention (Inspired by RWKV)

-   **ที่มา:** RWKV Architecture
-   **คำอธิบาย:** ใช้ Linear Attention ซึ่งมีความซับซ้อนเชิงเส้น (linear complexity) แทน Quadratic Attention แบบดั้งเดิม ทำให้ประมวลผล sequence ยาวๆ ได้มีประสิทธิภาพมากขึ้นทั้งในแง่ความเร็วและหน่วยความจำ
-   **การนำไปใช้:** `deepseek_utils.py` -> `apply_rwkv_linear_attention()` (เป็นการจำลองแนวคิด อาจไม่ใช่ implementation ตรง)
-   **วิธีเปิดใช้งาน:** `--use_rwkv` ใน `run_advanced_training.bat`

## 6. LLaMA-3 MoE (Inspired by LLaMA-3)

-   **ที่มา:** แนวคิด Mixture of Experts จาก LLaMA-3 และโมเดลอื่นๆ
-   **คำอธิบาย:** ใช้แนวคิด MoE โดยให้มี "experts" หลายส่วนในโมเดล และมี gating network คอยเลือกว่าจะใช้ expert ใดในการประมวลผล input แต่ละส่วน ช่วยเพิ่ม capacity ของโมเดลโดยไม่ต้องเพิ่มภาระการคำนวณมากนัก
-   **การนำไปใช้:** `deepseek_utils.py` -> `apply_llama3_mix_of_experts()` (เป็นการจำลองแนวคิด)
-   **วิธีเปิดใช้งาน:** `--use_llama3` ใน `run_advanced_training.bat`

## 7. Phi-3 GQA (Inspired by Phi-3)

-   **ที่มา:** Microsoft Phi-3
-   **คำอธิบาย:** Grouped-Query Attention (GQA) เป็นรูปแบบหนึ่งของ Multi-Query Attention ที่ให้หลาย query heads แชร์ key และ value heads ชุดเดียวกัน ช่วยลดภาระการคำนวณและหน่วยความจำในส่วน attention mechanism
-   **การนำไปใช้:** `deepseek_utils.py` -> `apply_phi3_grouped_query_attention()` (เป็นการจำลองแนวคิด)
-   **วิธีเปิดใช้งาน:** `--use_llama3` ใน `run_advanced_training.bat` (รวมอยู่กับ LLaMA-3 techniques)

## 8. Alpaca/Claude 3.5 Training Refinements

-   **ที่มา:** แนวปฏิบัติจาก Alpaca, Claude 3.5, และงานวิจัยอื่นๆ
-   **คำอธิบาย:** การปรับปรุงเล็กๆ น้อยๆ ในระหว่างการเทรน เช่น การใช้ learning rate schedule ที่เหมาะสม, การปรับปรุง data collation, การใช้เทคนิค regularization บางอย่าง เพื่อเพิ่มความเสถียรและประสิทธิภาพของการเทรน
-   **การนำไปใช้:** `deepseek_utils.py` -> `apply_alpaca_training_refinements()`, `apply_claude35_training_techniques()`
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ Multi-stage Training workflow
