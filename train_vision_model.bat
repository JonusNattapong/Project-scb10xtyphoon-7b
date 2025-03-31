@echo off
setlocal enabledelayedexpansion

echo ================================
echo Vision Model Training
echo ================================

echo Selecting training configuration...
echo 1. Basic Training (Default settings)
echo 2. Advanced Training (All techniques enabled)
echo 3. Custom Training (Select techniques)
set /p CONFIG_CHOICE="Select configuration (1-3): "

if "%CONFIG_CHOICE%"=="1" (
    python vision_training.py ^
        --learning_rate 1e-5 ^
        --num_epochs 100 ^
        --batch_size 4
) else if "%CONFIG_CHOICE%"=="2" (
    python vision_training.py ^
        --learning_rate 1e-5 ^
        --num_epochs 100 ^
        --batch_size 4 ^
        --use_flash_attention ^
        --enable_xformers ^
        --use_8bit_adam ^
        --use_token_merging ^
        --enable_selective_state_updates
) else if "%CONFIG_CHOICE%"=="3" (
    echo Select techniques to enable:
    
    set /p FLASH_ATTN="Use Flash Attention? (y/n): "
    set /p XFORMERS="Enable xFormers? (y/n): "
    set /p TOKEN_MERGE="Use Token Merging? (y/n): "
    set /p SEL_UPDATES="Enable Selective State Updates? (y/n): "
    set /p EIGHT_BIT="Use 8-bit Adam? (y/n): "
    
    set CMD=python vision_training.py --learning_rate 1e-5 --num_epochs 100 --batch_size 4
    if "!FLASH_ATTN!"=="y" set CMD=!CMD! --use_flash_attention
    if "!XFORMERS!"=="y" set CMD=!CMD! --enable_xformers
    if "!TOKEN_MERGE!"=="y" set CMD=!CMD! --use_token_merging
    if "!SEL_UPDATES!"=="y" set CMD=!CMD! --enable_selective_state_updates
    if "!EIGHT_BIT!"=="y" set CMD=!CMD! --use_8bit_adam
    
    !CMD!
) else (
    echo Invalid choice. Using basic configuration.
    python vision_training.py ^
        --learning_rate 1e-5 ^
        --num_epochs 100 ^
        --batch_size 4
)

echo.
echo Training completed! Check logs for details.
pause
