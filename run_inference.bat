@echo off
setlocal enabledelayedexpansion

REM Configuration
set MODEL_PATH=./fine_tuned_typhoon
set PEFT_MODEL_PATH=

REM Check if the model path exists
if not exist "%MODEL_PATH%" (
    echo Model path %MODEL_PATH% does not exist.
    echo Please provide a valid model path with the --model_path argument
    goto :usage
)

REM Parse command line arguments
set INTERACTIVE=0
set INPUT_FILE=
set OUTPUT_FILE=responses.txt
set SYSTEM_PROMPT=คุณเป็น AI ผู้ช่วยที่ฉลาดและมีประโยชน์ ตอบคำถามให้ได้ใจความกระชับและถูกต้อง
set TEMPERATURE=0.7
set MAX_LENGTH=512

:parse_args
if "%~1"=="" goto :run
if /i "%~1"=="--model_path" (
    set MODEL_PATH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--peft_model_path" (
    set PEFT_MODEL_PATH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--interactive" (
    set INTERACTIVE=1
    shift
    goto :parse_args
)
if /i "%~1"=="--input_file" (
    set INPUT_FILE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--output_file" (
    set OUTPUT_FILE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--system_prompt" (
    set SYSTEM_PROMPT=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--temperature" (
    set TEMPERATURE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--max_length" (
    set MAX_LENGTH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    goto :usage
)
echo Unknown argument: %~1
goto :usage

:usage
echo Usage: run_inference.bat [options]
echo Options:
echo   --model_path PATH        Path to the model (default: ./fine_tuned_typhoon)
echo   --peft_model_path PATH   Path to PEFT adapter (optional)
echo   --interactive            Run in interactive mode
echo   --input_file FILE        Input file with prompts
echo   --output_file FILE       Output file for responses (default: responses.txt)
echo   --system_prompt PROMPT   System prompt for interactive mode
echo   --temperature VALUE      Temperature for generation (default: 0.7)
echo   --max_length VALUE       Maximum response length (default: 512)
echo   --help                   Show this help message
exit /b 1

:run
REM Check if either interactive or input_file is specified
if %INTERACTIVE%==0 (
    if "%INPUT_FILE%"=="" (
        echo Either --interactive or --input_file must be specified
        goto :usage
    )
)

REM Prepare command
set CMD=python typhoon_inference.py --model_path "%MODEL_PATH%"

if not "%PEFT_MODEL_PATH%"=="" (
    set CMD=!CMD! --peft_model_path "%PEFT_MODEL_PATH%"
)

if %INTERACTIVE%==1 (
    set CMD=!CMD! --interactive
)

if not "%INPUT_FILE%"=="" (
    set CMD=!CMD! --input_file "%INPUT_FILE%"
)

if not "%OUTPUT_FILE%"=="" (
    set CMD=!CMD! --output_file "%OUTPUT_FILE%"
)

if not "%SYSTEM_PROMPT%"=="" (
    set CMD=!CMD! --system_prompt "%SYSTEM_PROMPT%"
)

set CMD=!CMD! --temperature %TEMPERATURE% --max_length %MAX_LENGTH%

REM Run the command
echo Running: !CMD!
!CMD!

endlocal
