# Advanced Vision Training Techniques

เอกสารนี้อธิบายเทคนิคขั้นสูงที่ใช้ในการเทรนโมเดล Vision (Stable Diffusion XL base) ในโปรเจกต์นี้

## 1. Flash Attention (Meta AI)

-   **ที่มา:** Paper "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
-   **คำอธิบาย:** การ implement attention mechanism ที่มีประสิทธิภาพสูง ลดการใช้หน่วยความจำและเพิ่มความเร็วในการคำนวณ โดยเฉพาะบน GPU รุ่นใหม่ๆ
-   **การนำไปใช้:** `vision_training.py` -> `VisionTrainer.__init__()` (ใช้ `to_bettertransformer()` หรือ integration ใน `diffusers`)
-   **วิธีเปิดใช้งาน:** `--use_flash_attention` ใน `train_vision_model.bat` (เปิดใช้งานโดย default ถ้า hardware รองรับ)

## 2. Advanced Schedulers (Anthropic, Stability AI)

-   **ที่มา:** DDPMScheduler (Anthropic), DPMSolverMultistepScheduler (Stability AI)
-   **คำอธิบาย:** ใช้ noise schedulers ที่ปรับปรุงแล้วเพื่อควบคุมกระบวนการ diffusion ให้มีประสิทธิภาพและคุณภาพผลลัพธ์ที่ดีขึ้น
    -   **DDPM (Denoising Diffusion Probabilistic Models):** Scheduler พื้นฐานที่ปรับปรุงโดย Anthropic
    -   **DPM++ 2M (Diffusion Probabilistic Model Solver++ 2nd Order Multistep):** Scheduler ขั้นสูงจาก Stability AI ที่ให้ผลลัพธ์คุณภาพสูงในจำนวน steps ที่น้อยลง
-   **การนำไปใช้:** `vision_training.py` -> `VisionTrainer.__init__()` (โหลด scheduler) และ `VisionTrainer.train()` (ใช้ในการ add noise และ inference)
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ `VisionTrainer` โดย default

## 3. Improved VAE (DeepMind, Stability AI)

-   **ที่มา:** "madebyollin/sdxl-vae-fp16-fix" (ปรับปรุงจาก VAE เดิม)
-   **คำอธิบาย:** ใช้ Variational Autoencoder (VAE) ที่ปรับปรุงแล้ว (เช่น แก้ปัญหา fp16 instability) เพื่อแปลงภาพไปมาระหว่าง pixel space และ latent space ได้อย่างมีประสิทธิภาพและแม่นยำมากขึ้น
-   **การนำไปใช้:** `vision_training.py` -> `VisionTrainer.__init__()` (โหลด VAE) และ `VisionTrainer.train()` (ใช้ encode ภาพเป็น latent)
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ `VisionTrainer` โดย default

## 4. Token Merging (EleutherAI)

-   **ที่มา:** Paper "Token Merging: Your ViT But Faster"
-   **คำอธิบาย:** เทคนิคสำหรับ Vision Transformer (ViT) ที่รวม token ที่มีความคล้ายคลึงกันใน layer ลึกๆ เพื่อลดจำนวน token ที่ต้องประมวลผลใน attention mechanism ช่วยเพิ่มความเร็วและลดการใช้หน่วยความจำ
-   **การนำไปใช้:** `vision_training.py` -> `VisionTrainer._setup_token_merging()` (เรียก `unet.enable_token_merging()`)
-   **วิธีเปิดใช้งาน:** `--use_token_merging` ใน `train_vision_model.bat`

## 5. Selective State Updates (Microsoft)

-   **ที่มา:** แนวคิดจากงานวิจัยของ Microsoft (คล้ายกับเทคนิคใน Phi-3)
-   **คำอธิบาย:** อัปเดตเฉพาะส่วนของ state ในโมเดลที่มีความสำคัญต่อการเรียนรู้ในแต่ละ step แทนที่จะอัปเดตทั้งหมด ช่วยลดภาระการคำนวณและอาจช่วยให้การเรียนรู้มีประสิทธิภาพมากขึ้น
-   **การนำไปใช้:** `vision_training.py` -> `VisionTrainer._setup_selective_state_updates()` (เรียก `unet.enable_selective_state_updates()`)
-   **วิธีเปิดใช้งาน:** `--enable_selective_state_updates` ใน `train_vision_model.bat`

## 6. 8-bit Adam Optimizer (Meta AI)

-   **ที่มา:** `bitsandbytes` library
-   **คำอธิบาย:** ใช้ Adam optimizer เวอร์ชั่นที่เก็บ state ในรูปแบบ 8-bit แทน 32-bit ช่วยลดการใช้หน่วยความจำของ optimizer ลงอย่างมาก (ประมาณ 4 เท่า) ทำให้สามารถเทรนโมเดลขนาดใหญ่ขึ้นหรือใช้ batch size ใหญ่ขึ้นได้
-   **การนำไปใช้:** `vision_training.py` -> `VisionTrainer.train()` (เลือก optimizer class)
-   **วิธีเปิดใช้งาน:** `--use_8bit_adam` ใน `train_vision_model.bat` (เปิดใช้งานโดย default ถ้าติดตั้ง `bitsandbytes` สำเร็จ)

## 7. Constitutional AI Checks (Anthropic)

-   **ที่มา:** Anthropic's Constitutional AI principles
-   **คำอธิบาย:** แนวคิดในการกรองข้อมูลหรือตรวจสอบผลลัพธ์เพื่อป้องกันการสร้างเนื้อหาที่เป็นอันตราย ไม่เหมาะสม หรือละเมิดหลักการที่กำหนดไว้ (Constitution)
-   **การนำไปใช้:** `vision_training.py` -> `VisionTrainer._constitutional_ai_check()` (เป็น placeholder สำหรับ implement safety checks)
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ training loop (ปัจจุบันยังไม่มี logic การกรองจริง)

## 8. Distributed Training (Accelerate)

-   **ที่มา:** Hugging Face Accelerate Library
-   **คำอธิบาย:** ช่วยให้การเทรนโมเดลบนหลาย GPU หรือหลายเครื่อง (multi-node) ทำได้ง่ายขึ้น จัดการการกระจายข้อมูลและโมเดลโดยอัตโนมัติ
-   **การนำไปใช้:** ทั่วทั้ง `vision_training.py` (ใช้ `Accelerator` object) และสคริปต์ launch (`train_vision_model.bat`)
-   **วิธีเปิดใช้งาน:** รันผ่าน `accelerate launch` (ทำโดยอัตโนมัติใน `.bat` script) และ config ผ่าน `accelerate config`
