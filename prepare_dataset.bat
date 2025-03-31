@echo off
setlocal enabledelayedexpansion

echo ================================
echo SCB10X Typhoon Dataset Preparer
echo ================================

:menu
echo.
echo เลือกประเภทข้อมูลที่ต้องการเตรียม:
echo 1) ข้อความทั่วไป (Text)
echo 2) บทสนทนา (Conversation)
echo 3) คำสั่งและผลลัพธ์ (Instruction)
echo 4) ข้อมูลภาพและคำบรรยาย (Vision)
echo 5) เรียนรู้แบบครูนักเรียน (Teacher-Student)
echo 6) ออกจากโปรแกรม

set /p choice="กรุณาเลือกหมายเลข (1-6): "

if "%choice%"=="1" goto text_dataset
if "%choice%"=="2" goto conversation_dataset
if "%choice%"=="3" goto instruction_dataset
if "%choice%"=="4" goto vision_dataset
if "%choice%"=="5" goto teacher_student_dataset
if "%choice%"=="6" goto end

echo เลือกไม่ถูกต้อง กรุณาลองใหม่
goto menu

:vision_dataset
echo เตรียมไฟล์ CSV ที่มีคอลัมน์:
echo - image: พาธไปยังไฟล์ภาพ
echo - text: ข้อความที่เกี่ยวข้อง (ถ้ามี)
echo - caption: คำบรรยายภาพ (ถ้ามี)
set /p input_file="ป้อนที่อยู่ไฟล์ข้อมูล (.csv): "
set /p output_dir="ป้อนโฟลเดอร์เก็บผลลัพธ์: "
set /p processor="ป้อนชื่อ Vision Processor (เช่น openai/clip-vit-base-patch32): "

python data_preprocessing.py ^
--input_file "%input_file%" ^
--input_type csv ^
--output_dir "%output_dir%" ^
--dataset_type vision ^
--processor_name "%processor%" ^
--image_column "image" ^
--text_column "text" ^
--caption_column "caption"

echo.
echo ดำเนินการเสร็จสิ้น! ตรวจสอบผลลัพธ์ได้ที่: %output_dir%
echo ตัวอย่างข้อมูลอยู่ในไฟล์: %output_dir%\sample_texts.txt
pause
goto menu

:teacher_student_dataset
echo เตรียมไฟล์ข้อความสำหรับการเรียนรู้แบบครูนักเรียน
echo โดยใช้ microsoft/OmniParser-v2.0 เป็นโมเดลครู
set /p input_file="ป้อนที่อยู่ไฟล์ข้อความ (.txt): "
set /p output_dir="ป้อนโฟลเดอร์เก็บผลลัพธ์: "

python data_preprocessing.py ^
--input_file "%input_file%" ^
--input_type text ^
--output_dir "%output_dir%" ^
--dataset_type teacher_student ^
--text_column text

echo.
echo ดำเนินการเสร็จสิ้น! ตรวจสอบผลลัพธ์ได้ที่: %output_dir%
echo ตัวอย่างข้อมูลอยู่ในไฟล์: %output_dir%\sample_texts.txt
pause
goto menu

:text_dataset
set /p input_file="ป้อนที่อยู่ไฟล์ข้อความ (.txt): "
set /p output_dir="ป้อนโฟลเดอร์เก็บผลลัพธ์: "

python data_preprocessing.py ^
--input_file "%input_file%" ^
--input_type text ^
--output_dir "%output_dir%" ^
--dataset_type text ^
--text_column text

echo.
echo ดำเนินการเสร็จสิ้น! ตรวจสอบผลลัพธ์ได้ที่: %output_dir%
echo ตัวอย่างข้อมูลอยู่ในไฟล์: %output_dir%\sample_texts.txt
pause
goto menu

:conversation_dataset
echo เตรียมไฟล์ CSV ที่มี columns: question, answer
set /p input_file="ป้อนที่อยู่ไฟล์บทสนทนา (.csv): "
set /p output_dir="ป้อนโฟลเดอร์เก็บผลลัพธ์: "

python data_preprocessing.py ^
--input_file "%input_file%" ^
--input_type csv ^
--output_dir "%output_dir%" ^
--dataset_type conversation

echo.
echo ดำเนินการเสร็จสิ้น! ตรวจสอบผลลัพธ์ได้ที่: %output_dir%
echo ตัวอย่างข้อมูลอยู่ในไฟล์: %output_dir%\sample_texts.txt
pause
goto menu

:instruction_dataset
echo เตรียมไฟล์ JSON ที่มี fields: instruction, input (optional), output
set /p input_file="ป้อนที่อยู่ไฟล์คำสั่ง (.json): "
set /p output_dir="ป้อนโฟลเดอร์เก็บผลลัพธ์: "

python data_preprocessing.py ^
--input_file "%input_file%" ^
--input_type json ^
--output_dir "%output_dir%" ^
--dataset_type instruction

echo.
echo ดำเนินการเสร็จสิ้น! ตรวจสอบผลลัพธ์ได้ที่: %output_dir%
echo ตัวอย่างข้อมูลอยู่ในไฟล์: %output_dir%\sample_texts.txt
pause
goto menu

:end
echo ปิดโปรแกรม...
exit /b 0
