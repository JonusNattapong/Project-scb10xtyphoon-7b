# API Documentation

## Data Preprocessing API

### `clean_text(text: str, keep_newlines: bool = True) -> str`
ทำความสะอาดข้อความโดยลบช่องว่างที่ไม่จำเป็นและจัดการขึ้นบรรทัดใหม่

Parameters:
- `text`: ข้อความที่ต้องการทำความสะอาด
- `keep_newlines`: เก็บการขึ้นบรรทัดใหม่หรือไม่ (default: True)

Returns:
- ข้อความที่ผ่านการทำความสะอาด

### `encode_image(image_path: str) -> str | None`
แปลงไฟล์ภาพเป็น base64 string

Parameters:
- `image_path`: พาธไปยังไฟล์ภาพ

Returns:
- base64 string ของภาพ หรือ None ถ้าเกิดข้อผิดพลาด

### `prepare_vision_dataset(dataset, image_column="image", text_column="text", caption_column="caption")`
เตรียมข้อมูลสำหรับ vision dataset

Parameters:
- `dataset`: Dataset object
- `image_column`: ชื่อคอลัมน์ที่เก็บพาธของภาพ
- `text_column`: ชื่อคอลัมน์ที่เก็บข้อความ
- `caption_column`: ชื่อคอลัมน์ที่เก็บคำบรรยายภาพ

Returns:
- Dataset ที่ผ่านการประมวลผลแล้ว

### `prepare_conversation_dataset(dataset, format_template="User: {question}\nAssistant: {answer}", image_column=None)`
เตรียมข้อมูลสำหรับ conversation dataset

Parameters:
- `dataset`: Dataset object
- `format_template`: รูปแบบของบทสนทนา
- `image_column`: ชื่อคอลัมน์ที่เก็บพาธของภาพ (optional)

Returns:
- Dataset ที่ผ่านการประมวลผลแล้ว

### `generate_teacher_outputs(dataset, teacher_model_name="microsoft/OmniParser-v2.0")`
สร้าง outputs จากโมเดลครูสำหรับ knowledge distillation

Parameters:
- `dataset`: Dataset object
- `teacher_model_name`: ชื่อโมเดลครูที่ใช้

Returns:
- Dataset ที่มี teacher outputs

## Command Line Interface

### prepare_dataset.bat
```bash
prepare_dataset.bat
```
เครื่องมือ CLI สำหรับเตรียมข้อมูล มีตัวเลือก:
1. Text Dataset
2. Conversation Dataset
3. Instruction Dataset
4. Vision Dataset
5. Teacher-Student Dataset

## File Formats

### Text Dataset
```text
ข้อความบรรทัดที่ 1
ข้อความบรรทัดที่ 2
...
```

### Conversation Dataset (CSV)
```csv
question,answer
"คำถาม 1","คำตอบ 1"
"คำถาม 2","คำตอบ 2"
...
```

### Vision Dataset (CSV)
```csv
image,text,caption
"path/to/image1.jpg","ข้อความ 1","คำบรรยาย 1"
"path/to/image2.jpg","ข้อความ 2","คำบรรยาย 2"
...
```

### Instruction Dataset (JSON)
```json
{
  "instructions": [
    {
      "instruction": "คำสั่ง",
      "input": "อินพุต (ถ้ามี)",
      "output": "เอาต์พุต"
    }
  ]
}
```

## Error Handling

ระบบจะสร้างไฟล์ log (data_preprocessing.log) ที่บันทึก:
- การโหลดข้อมูล
- การประมวลผล
- ข้อผิดพลาดที่เกิดขึ้น
- สถิติต่างๆ

## Environment Variables

- `OPENWEATHER_API_KEY`: API key สำหรับ Vision Processor
- `CUDA_VISIBLE_DEVICES`: กำหนด GPU ที่ใช้
- `TOKENIZERS_PARALLELISM`: ควบคุมการทำงานแบบขนาน

## Image Generation API

### VisionEnhancer Class

```python
enhancer = VisionEnhancer(
    clip_model_name="openai/clip-vit-large-patch14",
    sdxl_model_name="stabilityai/stable-diffusion-xl-base-1.0",
    controlnet_model="lllyasviel/sd-controlnet-canny"
)
```

### การสร้างภาพ (Image Generation)

1. Text-to-Image:
```python
# สร้างภาพจากข้อความ
image = enhancer.text_to_image(
    prompt="Ancient Thai temple with golden spires",
    negative_prompt="modern buildings, poor quality",
    style_preset="photographic",  # หรือ "artistic", "anime"
    width=1024,
    height=1024,
    guidance_scale=7.5
)
```

2. Image-and-Text-to-Image:
```python
# สร้างภาพใหม่จากภาพและข้อความ
source = Image.open("temple.jpg")
new_image = enhancer.image_and_text_to_image(
    source_image=source,
    prompt="Add sunset lighting and dramatic sky",
    strength=0.8,  # ความเข้มของการปรับแต่ง
    guidance_scale=7.5
)
```

3. ControlNet:
```python
# สร้างภาพโดยใช้ภาพควบคุม
control = Image.open("sketch.jpg")
image = enhancer.control_image(
    prompt="Traditional Thai painting, detailed art",
    control_image=control,
    guidance_scale=7.5
)
```

### Style Presets

ตัวอย่าง style presets ที่ใช้ได้:
```python
styles = {
    "photographic": "professional photography, high quality, sharp focus",
    "artistic": "artistic painting, expressive, beautiful composition",
    "anime": "anime style, cel shading, vibrant colors",
    "watercolor": "watercolor painting, soft edges, flowing colors",
    "sketch": "detailed pencil sketch, fine lines, artistic drawing",
    "thai": "traditional Thai art style, gold leaf, intricate details"
}
```

### ถ่ายทอดสไตล์

```python
# ถ่ายทอดสไตล์ไปยังภาพ
content_image = Image.open("photo.jpg")
styled_image = enhancer.style_transfer(
    content_image=content_image,
    style_prompt="Traditional Thai painting style",
    strength=0.75
)
```

### ปรับปรุงภาพด้วย Prompt

```python
# ปรับปรุงภาพโดยใช้ prompt
image = Image.open("temple.jpg")
enhanced = enhancer.enhance_with_prompt(
    image=image,
    enhancement_prompt="Enhance architectural details, improve lighting",
    strength=0.5
)
```

## ตัวอย่าง Prompts

ดูตัวอย่าง prompts ได้ใน `example_datasets/vision_samples/prompts.json`:

### Image Generation
```json
{
  "name": "Thai Temple",
  "prompt": "Ancient Thai temple with golden spires...",
  "negative_prompt": "modern buildings, people...",
  "width": 1024,
  "height": 1024
}
```

### Style Transfer
```json
{
  "name": "Thai Art Style",
  "style_prompt": "Traditional Thai painting style...",
  "strength": 0.8
}
```

### Enhancement
```json
{
  "name": "Temple Enhancement",
  "prompt": "Enhance architectural details...",
  "strength": 0.6
}
```

### บันทึกและโหลดโมเดล

```python
# บันทึกโมเดลในรูปแบบ safetensors (แนะนำ)
enhancer.save_model(
    path="models/vision_model",
    use_safetensors=True  # default: True
)

# โหลดโมเดลจาก safetensors
enhancer.load_model(
    path="models/vision_model",
    use_safetensors=True  # default: True
)

# บันทึกในรูปแบบ PyTorch
enhancer.save_model(
    path="models/vision_model",
    use_safetensors=False
)
```

ข้อดีของ Safetensors:
- ปลอดภัยกว่า: ไม่มีความเสี่ยงจาก pickle
- เร็วกว่า: โหลดไฟล์ได้เร็วกว่า pickle
- ประหยัดพื้นที่: ขนาดไฟล์เล็กกว่า
- แชร์ง่าย: สามารถใช้ข้ามแพลตฟอร์มได้

## Dependencies

ดูรายละเอียดใน requirements.txt:
- transformers
- torch
- pandas
- numpy
- pillow
- tqdm
- diffusers
- accelerate
- safetensors
