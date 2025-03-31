@echo off
setlocal enabledelayedexpansion

echo ================================
echo Advanced Text Model Training
echo ================================

echo Configuring Accelerate...
accelerate config default

set BASE_CMD=accelerate launch advanced_training.py
set ARGS=

:parse_args
if "%1"=="" goto end_parse
if "%1"=="--output_dir" (
    set ARGS=!ARGS! --output_dir %2
    shift /1
    shift /1
) else if "%1"=="--cache_dir" (
    set ARGS=!ARGS! --cache_dir %2
    shift /1
    shift /1
) else if "%1"=="--model_path" (
    set ARGS=!ARGS! --model_path %2
    shift /1
    shift /1
) else if "%1"=="--quantization" (
    set ARGS=!ARGS! --quantization %2
    shift /1
    shift /1
) else if "%1"=="--peft_method" (
    set ARGS=!ARGS! --peft_method %2
    shift /1
    shift /1
) else if "%1"=="--resume_from_checkpoint" (
    set ARGS=!ARGS! --resume_from_checkpoint %2
    shift /1
    shift /1
) else if "%1"=="--use_flash_attn" (
    set ARGS=!ARGS! --use_flash_attn
    shift /1
) else if "%1"=="--use_rwkv" (
    set ARGS=!ARGS! --use_rwkv
    shift /1
) else if "%1"=="--use_llama3" (
    set ARGS=!ARGS! --use_llama3
    shift /1
) else (
    echo Unknown argument: %1
    shift /1
)
goto parse_args

:end_parse

echo Running command: %BASE_CMD% %ARGS%
%BASE_CMD% %ARGS%

echo.
echo Training completed! Check logs for details.
pause
