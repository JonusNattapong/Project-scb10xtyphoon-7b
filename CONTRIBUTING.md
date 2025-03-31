# Contributing Guidelines

ขอขอบคุณที่สนใจร่วมพัฒนา SCB10X Typhoon-7B! นี่คือแนวทางในการมีส่วนร่วมพัฒนาโครงการ

## การรายงานปัญหา (Bug Reports)

1. ตรวจสอบว่าปัญหานี้ได้ถูกรายงานแล้วหรือไม่ใน Issues
2. ใช้ template ที่กำหนดในการรายงานปัญหา
3. ให้ข้อมูลที่จำเป็นครบถ้วน:
   - ขั้นตอนการทำให้เกิดปัญหา
   - ผลลัพธ์ที่คาดหวัง vs ผลลัพธ์ที่ได้
   - Log files หรือ error messages
   - สภาพแวดล้อมที่ใช้ (OS, Python version, etc.)

## การเสนอฟีเจอร์ใหม่ (Feature Requests)

1. เปิด Issue ใหม่พร้อมระบุ label "enhancement"
2. อธิบายฟีเจอร์ที่ต้องการและเหตุผล
3. หากเป็นไปได้ ให้เสนอแนวทางการพัฒนาคร่าวๆ

## การส่ง Pull Requests

1. Fork โปรเจคและสร้าง branch ใหม่
2. ทำการพัฒนาใน branch ของคุณ
3. เขียน tests ให้ครอบคลุมฟีเจอร์ใหม่
4. รัน tests ทั้งหมดให้ผ่าน
5. ส่ง Pull Request พร้อมอธิบายการเปลี่ยนแปลง

### Coding Guidelines

1. ใช้ Black formatter สำหรับ Python code
2. เขียน docstring สำหรับทุก function และ class
3. ตั้งชื่อตัวแปรและฟังก์ชันให้สื่อความหมาย
4. เพิ่ม comments อธิบายโค้ดที่ซับซ้อน
5. อัพเดท documentation เมื่อมีการเปลี่ยนแปลง API

### Commit Messages

- ใช้ภาษาอังกฤษ
- ขึ้นต้นด้วย verb (add, fix, update, etc.)
- บรรทัดแรกไม่เกิน 50 ตัวอักษร
- อธิบายรายละเอียดในบรรทัดถัดไป

ตัวอย่าง:
```
Add vision dataset processing functionality

- Add support for image preprocessing
- Implement base64 encoding
- Add example vision dataset
- Update documentation
```

## Documentation

1. อัพเดท README.md ถ้ามีการเปลี่ยนแปลงที่สำคัญ
2. เพิ่มตัวอย่างการใช้งานสำหรับฟีเจอร์ใหม่
3. อัพเดท requirements.txt ถ้ามีการเพิ่ม dependencies

## Testing

1. เขียน unit tests สำหรับฟีเจอร์ใหม่
2. ตรวจสอบ code coverage
3. รัน integration tests
4. ทดสอบ edge cases

## Review Process

1. PR จะถูก review โดยทีมพัฒนา
2. อาจมีการขอให้แก้ไขหรือเพิ่มเติม
3. PR จะถูก merge เมื่อผ่านการ review และ tests ทั้งหมด

## สอบถามเพิ่มเติม

- เปิด Issue สำหรับคำถามทั่วไป
- ติดต่อทีมพัฒนาผ่าน email: dev@scb10x.com
