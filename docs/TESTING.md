# Testing Guide

## การทดสอบระบบ

### 1. Unit Tests

ใช้ pytest สำหรับทดสอบ components ต่างๆ:

```bash
# รัน unit tests ทั้งหมด
pytest tests/

# รันเฉพาะ test บางส่วน
pytest tests/test_data_preprocessing.py
pytest tests/test_vision_processing.py
```

#### Test Cases ที่สำคัญ

1. Data Preprocessing:
```python
def test_clean_text():
    # Test normal text
    assert clean_text("hello   world") == "hello world"
    
    # Test newlines
    assert clean_text("line1\n\nline2") == "line1\nline2"
    
    # Test Thai text
    assert clean_text("สวัสดี   ครับ") == "สวัสดี ครับ"
```

2. Vision Processing:
```python
def test_encode_image():
    # Test valid image
    result = encode_image("test_image.jpg")
    assert result is not None
    assert isinstance(result, str)
    
    # Test invalid image
    assert encode_image("nonexistent.jpg") is None
```

3. Teacher-Student:
```python
def test_teacher_outputs():
    dataset = create_test_dataset()
    result = generate_teacher_outputs(dataset)
    assert "teacher_logits" in result.features
```

### 2. Integration Tests

ทดสอบการทำงานร่วมกันของ components:

1. Data Pipeline Test:
```python
def test_full_pipeline():
    # Test text data
    text_result = process_text_dataset(sample_text)
    assert text_result is not None
    
    # Test vision data
    vision_result = process_vision_dataset(sample_vision)
    assert vision_result is not None
    
    # Test combined
    combined = process_combined_dataset(text_result, vision_result)
    assert combined is not None
```

2. Model Training Test:
```python
def test_model_training():
    # Prepare test data
    train_data = prepare_test_data()
    
    # Test training
    model = train_model(train_data)
    assert model is not None
    
    # Test inference
    result = model.generate("Test prompt")
    assert result is not None
```

### 3. Performance Tests

1. Memory Usage:
```python
def test_memory_usage():
    # Track memory before
    mem_before = get_memory_usage()
    
    # Run process
    process_large_dataset()
    
    # Track memory after
    mem_after = get_memory_usage()
    
    # Check memory leak
    assert (mem_after - mem_before) < MEMORY_THRESHOLD
```

2. Processing Speed:
```python
def test_processing_speed():
    start_time = time.time()
    process_dataset(large_dataset)
    end_time = time.time()
    
    processing_time = end_time - start_time
    assert processing_time < TIME_THRESHOLD
```

### 4. Edge Cases

1. Empty Input:
```python
def test_empty_input():
    assert clean_text("") == ""
    assert encode_image("") is None
    assert process_dataset([]) is not None
```

2. Thai Character Edge Cases:
```python
def test_thai_characters():
    # Test combining characters
    assert clean_text("กิ์") == "กิ์"
    
    # Test tone marks
    assert clean_text("ก่") == "ก่"
```

3. Large Input:
```python
def test_large_input():
    # Create large text
    large_text = "า" * 1000000
    
    # Should not crash
    result = clean_text(large_text)
    assert result is not None
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/
```

## Test Coverage

ใช้ pytest-cov สำหรับวัด coverage:

```bash
pytest --cov=src tests/
```

เป้าหมาย coverage:
- Unit Tests: >90%
- Integration Tests: >80%
- Overall: >85%

## การเขียน Test Cases ใหม่

1. ตั้งชื่อ test function ให้สื่อความหมาย:
```python
def test_should_clean_thai_text_with_special_characters():
    ...
```

2. ใช้ pytest fixtures:
```python
@pytest.fixture
def sample_dataset():
    return create_test_dataset()

def test_processing(sample_dataset):
    result = process_dataset(sample_dataset)
    assert result is not None
```

3. Group tests ด้วย classes:
```python
class TestDataPreprocessing:
    def test_clean_text(self):
        ...
    
    def test_encode_image(self):
        ...
```

## Automated Testing

1. Local Testing:
```bash
# Install pre-commit hooks
pre-commit install

# Run tests before commit
pre-commit run --all-files
```

2. CI/CD Pipeline:
- GitHub Actions จะรัน tests อัตโนมัติเมื่อมี push หรือ pull request
- สร้าง test report และ coverage report
- แจ้งเตือนเมื่อ tests fail
