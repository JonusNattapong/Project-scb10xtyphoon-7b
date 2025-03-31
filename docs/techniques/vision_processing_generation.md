# Vision Processing & Generation Techniques

เอกสารนี้อธิบายเทคนิคที่ใช้ในการประมวลผลและสร้างภาพใน `vision_utils.py`

## 1. CLIP (Contrastive Language–Image Pre-training)

-   **ที่มา:** OpenAI
-   **คำอธิบาย:** โมเดลที่เรียนรู้ความสัมพันธ์ระหว่างภาพและข้อความ ทำให้สามารถ:
    -   **สกัด Embeddings:** สร้าง vector representation ของทั้งภาพและข้อความที่อยู่ใน space เดียวกัน
    -   **คำนวณ Similarity:** วัดความคล้ายคลึงกันระหว่างภาพและข้อความ (เช่น ภาพแมว กับ คำว่า "แมว")
-   **การนำไปใช้:**
    -   `VisionEnhancer.__init__()`: โหลด CLIP model และ processor
    -   `VisionEnhancer.extract_features()`: ใช้ CLIP สกัด image/text embeddings และคำนวณ similarity
-   **วิธีเปิดใช้งาน:** เป็นส่วนประกอบหลักของ `VisionEnhancer`

## 2. Image Enhancement Techniques

### 2.1 Real-ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)

-   **ที่มา:** Paper "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
-   **คำอธิบาย:** เทคนิค Super-Resolution ขั้นสูงที่สามารถเพิ่มความละเอียดของภาพ (upscaling) พร้อมทั้งปรับปรุงรายละเอียดและลด noise/artifacts ได้ดีเยี่ยม แม้กับภาพคุณภาพต่ำในโลกจริง
-   **การนำไปใช้:** `VisionEnhancer._apply_real_esrgan()` (เป็นการจำลองแนวคิด multi-scale processing ที่ได้แรงบันดาลใจมา)
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ `VisionEnhancer.enhance_image()`

### 2.2 HDR-Net (Inspired by HDR-Net)

-   **ที่มา:** Paper "Deep Bilateral Learning for Real-Time Image Enhancement"
-   **คำอธิบาย:** ใช้ Neural Network ในการเรียนรู้การปรับแต่งภาพแบบ local และ global เพื่อเพิ่ม Dynamic Range (ความต่างระหว่างส่วนมืดสุดและสว่างสุด) ทำให้ภาพดูมีมิติและสมจริงขึ้น
-   **การนำไปใช้:** `VisionEnhancer._apply_hdr_enhancement()` (จำลองแนวคิด local-global feature และ adaptive tone mapping)
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ `VisionEnhancer.enhance_image()`

### 2.3 NAFNet (Non-linear Activation Free Network)

-   **ที่มา:** Paper "Simple Baselines for Image Restoration"
-   **คำอธิบาย:** โครงสร้าง Network ที่เรียบง่ายแต่มีประสิทธิภาพสูงสำหรับการฟื้นฟูภาพ (Image Restoration) เช่น การลด noise, เพิ่มความคมชัด โดยไม่ต้องใช้ non-linear activation function ที่ซับซ้อน
-   **การนำไปใช้:** `VisionEnhancer._apply_nafnet()` (จำลองแนวคิด multi-stage enhancement และ residual block)
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ `VisionEnhancer.enhance_image()`

## 3. Vision Transformer (ViT) Feature Extraction

-   **ที่มา:** Paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
-   **คำอธิบาย:** สกัด features จากภาพโดยมองภาพเป็นลำดับของ patches (เหมือนคำในประโยค) แล้วใช้ Transformer architecture ประมวลผล ทำให้ได้ features ที่มีความหมายในระดับต่างๆ (hierarchical)
-   **การนำไปใช้:** `VisionEnhancer._extract_vit_features()` (ดึง hidden states จาก CLIP's vision model และ reshape/pool)
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ `VisionEnhancer.extract_features()`

## 4. Attention Map Generation (Inspired by DALL-E 3)

-   **ที่มา:** แนวคิดจาก DALL-E 3 และงานวิจัยด้าน attention visualization
-   **คำอธิบาย:** สร้างภาพที่แสดงให้เห็นว่าโมเดล (ในที่นี้คือ ViT ภายใน CLIP) ให้ความสนใจกับส่วนใดของภาพเมื่อประมวลผล ช่วยในการทำความเข้าใจการทำงานของโมเดล
-   **การนำไปใช้:** `VisionEnhancer._generate_attention_maps()` (ดึง attention weights จาก CLIP's vision model และนำมาเฉลี่ย/reshape)
-   **วิธีเปิดใช้งาน:** เป็นส่วนหนึ่งของ `VisionEnhancer.extract_features()`

## 5. Image Generation Techniques (Diffusers Library)

### 5.1 Stable Diffusion XL (SDXL)

-   **ที่มา:** Stability AI
-   **คำอธิบาย:** โมเดล Text-to-Image ขั้นสูงที่มีขนาดใหญ่และให้ผลลัพธ์คุณภาพสูง สามารถสร้างภาพตาม prompt ได้หลากหลายสไตล์
-   **การนำไปใช้:** `VisionEnhancer.__init__()` (โหลด `StableDiffusionXLPipeline`), `VisionEnhancer.text_to_image()`, `VisionEnhancer.image_and_text_to_image()`, `VisionEnhancer.style_transfer()`, `VisionEnhancer.enhance_with_prompt()`
-   **วิธีเปิดใช้งาน:** เป็นโมเดลหลักในการสร้างภาพ

### 5.2 ControlNet

-   **ที่มา:** Paper "Adding Conditional Control to Text-to-Image Diffusion Models"
-   **คำอธิบาย:** เพิ่มความสามารถในการควบคุมการสร้างภาพของ Diffusion Model โดยใช้ input เพิ่มเติม เช่น ภาพโครงร่าง (canny edge), ท่าทาง (pose), หรือ depth map ทำให้สร้างภาพได้ตรงตามโครงสร้างที่ต้องการมากขึ้น
-   **การนำไปใช้:** `VisionEnhancer.__init__()` (โหลด `ControlNetModel` และ `StableDiffusionControlNetPipeline`), `VisionEnhancer.control_image()`
-   **วิธีเปิดใช้งาน:** เรียกใช้ `control_image()` พร้อมระบุ `control_image`

### 5.3 Text-to-Image

-   **คำอธิบาย:** สร้างภาพขึ้นมาใหม่ทั้งหมดจากคำอธิบาย (prompt)
-   **การนำไปใช้:** `VisionEnhancer.text_to_image()`

### 5.4 Image-and-Text-to-Image (Image-to-Image with Prompt)

-   **คำอธิบาย:** ปรับแต่งภาพต้นฉบับตามคำอธิบาย (prompt) โดยยังคงโครงสร้างเดิมของภาพไว้บางส่วน (ควบคุมด้วย `strength`)
-   **การนำไปใช้:** `VisionEnhancer.image_and_text_to_image()`

### 5.5 Style Transfer

-   **คำอธิบาย:** คล้ายกับ Image-and-Text-to-Image แต่เน้นการถ่ายทอด "สไตล์" จาก prompt ไปยังภาพต้นฉบับ
-   **การนำไปใช้:** `VisionEnhancer.style_transfer()`
