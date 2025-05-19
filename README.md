# Dementia and Parkinson Diagnosis with Vbai Models

# Vbai-DPA 2.2 Version (EN)

| Model | Test Size | Params | FLOPs | mAPᵛᵃᴵ | CPU b1 | V100 b1 | V100 b32 |
|-------|-------|--------|-------|--------|--------|---------|----------|
| **Vbai-DPA 2.2f** | _448_ | 51.41 M | 0.60 B | %91.11 | 26.01 ms | 13.00 ms | 2.60 ms |
| **Vbai-DPA 2.2c** | _448_ | 205.62 M | 2.23 B | %91.11 | 148.68 ms | 74.34 ms | 14.87 ms |
| **Vbai-DPA 2.2q** | _448_ | 207.08 M | 11.65 B | %91.11 | 157.22 ms | 78.61 ms | 15.72 ms |

## Description

The Vbai-DPA 2.2 (Dementia, Parkinson, Alzheimer) model has been trained and developed to diagnose brain diseases through MRI or fMRI images. It shows whether the patient has Parkinson's disease, dementia status and Alzheimer's risk with high accuracy. According to Vbai-DPA 2.1, they are divided into three classes based on performance, and are fine-tuned and trained versions with more data.

#### Audience / Target

Vbai models are developed exclusively for hospitals, health centres and science centres.

### Classes

 - **Alzheimer's disease**: The sick person definitely has Alzheimer's disease.
 - **Average Risk of Alzheimer's Disease**: The sick person may develop Alzheimer's disease in the near future.
 - **Mild Alzheimer's Risk**: The sick person has a little more time to develop Alzheimer's disease.
 - **Very Mild Alzheimer's Risk**: The sick person has time to reach the level of Alzheimer's disease.
 - **No Risk**: The person does not have any risk.
 - **Parkinson's Disease**: The person has Parkinson's disease.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://miniature-eureka-g9qx5444xpfvvw-7860.app.github.dev/)

[![Open in Hugging Face Spaces](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/spaces/Neurazum/Vbai-DPA-2.2c)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
