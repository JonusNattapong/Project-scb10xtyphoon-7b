# Resource Estimation

เอกสารนี้ประเมินทรัพยากรที่จำเป็นสำหรับการใช้งานและพัฒนาโปรเจกต์ SCB10X Typhoon-7B Fine-tuning

## 1. Hardware Requirements

### Minimum Requirements (สำหรับการ Inference และ Fine-tuning พื้นฐาน)

-   **CPU:** Intel Core i5 (รุ่นใหม่) หรือเทียบเท่า
-   **RAM:** 16 GB
-   **GPU:** NVIDIA GPU พร้อม VRAM 12 GB (เช่น RTX 3060) และ CUDA Compute Capability 7.0+
-   **Storage:** 50 GB SSD (สำหรับ OS, dependencies, และโมเดลพื้นฐาน)

### Recommended Requirements (สำหรับการ Fine-tuning ขั้นสูง และ Vision Training)

-   **CPU:** Intel Core i7/i9 หรือ AMD Ryzen 7/9 (รุ่นใหม่)
-   **RAM:** 32 GB ขึ้นไป (64 GB+ สำหรับ Vision Training ที่ซับซ้อน)
-   **GPU:** NVIDIA GPU พร้อม VRAM 24 GB ขึ้นไป (เช่น RTX 3090, RTX 4090, A100)
    -   สำหรับ Distributed Training: แนะนำ GPU รุ่นเดียวกันหลายตัว
    -   รองรับ Flash Attention 2 (Ampere, Ada Lovelace, Hopper) เพื่อประสิทธิภาพสูงสุด
-   **Storage:** 200 GB+ NVMe SSD (สำหรับ datasets ขนาดใหญ่, checkpoints, และโมเดลหลายเวอร์ชัน)
-   **Network:** การเชื่อมต่ออินเทอร์เน็ตความเร็วสูง (สำหรับการดาวน์โหลด datasets และ models)

## 2. Software Requirements

-   **Operating System:** Linux (แนะนำ Ubuntu 20.04+), Windows 10/11 (พร้อม WSL2 สำหรับ Linux environment), macOS (อาจมีข้อจำกัดด้าน GPU)
-   **Python:** 3.8 - 3.10
-   **CUDA Toolkit:** 11.7 ขึ้นไป (ขึ้นอยู่กับเวอร์ชัน PyTorch และ GPU driver)
-   **cuDNN:** เวอร์ชั่นที่เข้ากันได้กับ CUDA
-   **NVIDIA Driver:** เวอร์ชั่นล่าสุดที่รองรับ CUDA Toolkit
-   **Git:** สำหรับ version control
-   **Package Manager:** `pip` หรือ `conda`
-   **Dependencies:** ดูรายละเอียดใน `requirements.txt` (หลักๆ คือ `torch`, `transformers`, `datasets`, `accelerate`, `diffusers`, `peft`, `bitsandbytes`, `safetensors`)

## 3. Data Requirements

### Download Size (ประมาณการ)

-   **Text Datasets (Wikipedia, ThaiGPT4):** ~5-10 GB
-   **Conversation Datasets (BELLE, Alpaca):** ~1-2 GB
-   **Vision Datasets (LAION Thai, Thai Art):** อาจมีขนาดใหญ่มาก ขึ้นอยู่กับ subset ที่โหลด (LAION Thai อาจถึง TBs, Thai Art ~10-20 GB)
-   **Instruction Datasets (ThaiInstruct, Dolly):** ~500 MB - 1 GB
-   **Total (ไม่รวม LAION เต็ม):** ~20-40 GB (อาจมากกว่านี้ขึ้นอยู่กับเวอร์ชัน dataset)

### Processing Space

-   ต้องการพื้นที่เพิ่มเติมสำหรับ datasets ที่ผ่านการประมวลผล (tokenized, formatted) อาจเพิ่มขึ้น 1.5x - 2x ของขนาดเดิม
-   Cache ของ `datasets` library อาจใช้พื้นที่พอสมควร

### Storage Recommendation

-   **สำหรับ Text Training:** แนะนำพื้นที่ว่างอย่างน้อย 100 GB
-   **สำหรับ Vision Training (รวม LAION subset):** แนะนำพื้นที่ว่างอย่างน้อย 500 GB - 1 TB+

## 4. Model Storage

-   **Base Models:**
    -   Typhoon-7B: ~14 GB (fp16)
    -   Stable Diffusion XL Base: ~7 GB (fp16)
    -   CLIP Model: ~1-2 GB
    -   ControlNet: ~1.5 GB
    -   VAE Fix: ~300 MB
-   **Fine-tuned Models / Checkpoints:**
    -   LoRA adapters: มีขนาดเล็ก (MBs)
    -   Full fine-tuned models: ขนาดใกล้เคียง base model
    -   Checkpoints ระหว่างเทรน: อาจใช้พื้นที่มาก ขึ้นอยู่กับความถี่ในการ save

### Storage Recommendation

-   **สำหรับ Text:** 50 GB+ สำหรับ base model และ checkpoints
-   **สำหรับ Vision:** 100 GB+ สำหรับ base models และ checkpoints

## 5. Training Time Estimation (ต่อ Epoch, ขึ้นอยู่กับ Hardware และ Dataset)

-   **Text Fine-tuning (SFT/DPO):**
    -   Single RTX 3090: ชั่วโมง - หลายชั่วโมง
    -   Multi-GPU (A100 x8): นาที - ชั่วโมง
-   **Vision Training (SDXL):**
    -   Single RTX 4090: หลายชั่วโมง - วัน
    -   Multi-GPU (A100 x8): ชั่วโมง - หลายชั่วโมง

**หมายเหตุ:** เวลาในการเทรนเป็นค่าประมาณการคร่าวๆ และขึ้นอยู่กับปัจจัยหลายอย่าง เช่น ขนาด dataset, batch size, learning rate, จำนวน epochs, และประสิทธิภาพของ hardware

## สรุป

การใช้งานโปรเจกต์นี้ โดยเฉพาะส่วน Vision Training และการเทรนขั้นสูง ต้องการทรัพยากร Hardware ค่อนข้างสูง โดยเฉพาะ GPU ที่มี VRAM ขนาดใหญ่ และพื้นที่จัดเก็บข้อมูลที่เพียงพอ การใช้เทคนิค Optimization เช่น QLoRA, 8-bit Adam, และ Distributed Training จะช่วยลดข้อจำกัดด้านทรัพยากรลงได้
