import data_struct.output_data_struct as ods
import ocr
import openai
import dotenv
import os
import base64
import re
import json

import pydantic as pdt
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

dotenv.load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'

raw_img_dir = './raw_dataset'
label_file = './file.jsonl'

oc = ocr.ocr()

client = openai.OpenAI(api_key = api_key, base_url = base_url)

# --- Pydantic Models ---

class CategoryType(StrEnum):
    DINING = 'Dining'
    TRANSP = 'Transportation'
    GROCER = 'Groceries'      
    SHOPIN = 'Shopping'      
    SERVCE = 'Services'     
    ACCOMO = 'Accommodation'
    ENTERT = 'Entertainment' 
    OTHERS = 'Others'       

class CurrencyType(StrEnum):
    HKD = 'HKD'  
    USD = 'USD' 
    JPY = 'JPY'
    CNY = 'CNY'
    TWD = 'TWD' 
    EUR = 'EUR'
    GBP = 'GBP'

class Bill(BaseModel):
    merchant:   str
    datetime:   datetime
    currency:   CurrencyType
    amount:     str = pdt.Field(pattern = r'^\d+\.\d{2}$|^\d+$') 
    category:   CategoryType

    @pdt.model_validator(mode='after')
    def validate_currency_precision(self):
        zero_decimal_currencies = {CurrencyType.JPY, CurrencyType.TWD}
        if self.currency in zero_decimal_currencies:
            if not re.match(r'^\d+$', self.amount): 
                raise ValueError(f"{self.currency} amounts cannot have decimal places (received {self.amount}).")
        else:
            # Fixed: Added self.amount as argument to re.match
            if not re.match(r'^\d+\.\d{2}$', self.amount):
                raise ValueError(f"{self.currency} only accept amount with 2 decimal places (received {self.amount}).")
        return self

# ---------------------------------------------------------------------------------------

system_prompt = \
"""You are now a receipt reader, and the receipt will be provided in the form of an image by the user. 
Try to extract the following information from the given image and output it as a JSON object:
- Total Amount Paid.
- The Merchant
- The Payment Method
- Paid Currency

Ensure the output is a valid JSON object matching this schema:
{
  "merchant": "string",
  "datetime": "YYYY-MM-DDTHH:MM:SS",
  "currency": "Enum(HKD, USD, JPY, CNY, TWD, EUR, GBP)",
  "amount": "string (formatted as '10.00' or '100')",
  "category": "Enum(Dining, Transportation, Groceries, Shopping, Services, Accommodation, Entertainment, Others)"
}
"""

def get_base64_img(r_img_dir, r_img):
    r_img_full_dir = os.path.join(r_img_dir, r_img)
    img_type = r_img.split('.')[-1]
    # Encode the image to base64 and get image type
    with open(r_img_full_dir, 'rb') as file:
        base64_img = base64.b64encode(file.read()).decode('utf-8')
    return base64_img, img_type

def get_gemini_response(system_prompt: str, img_type, base64_img):
    # Get Gemini Response
    try:
        print('sending request...')
        response = client.chat.completions.create(
            model = "gemini-3-flash-preview", 
            messages = [
                {
                    "role":"system",
                    "content": system_prompt, 
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Describe this image and extract data in JSON." 
                        },
                        {
                            "type": "image_url", 
                            "image_url":{"url": f"data:image/{img_type};base64,{base64_img}"}
                        },
                    ],
                },
            ],
        )
        print('request received')
        message = response.choices[0].message.content
        return message

    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return None

def get_ocr_result(r_img_dir, r_img):
    # This is blocking
    return oc.after_care(oc.infer(os.path.join(r_img_dir, r_img)))

def process_one_img(r_img_dir, r_img):
    # Get Base64
    print("claculating b64 image...")
    b64img, img_type = get_base64_img(r_img_dir, r_img)
    # Get OCR result
    print("obtaining OCR result...")
    ocr_result = get_ocr_result(r_img_dir, r_img), 
    print("obtaining VLM result...")
    label = get_gemini_response(system_prompt, img_type, b64img)
    return os.path.join(r_img_dir, r_img), ocr_result[0], label 

def main():
    if not os.path.exists(raw_img_dir):
        raise ValueError("target directory for raw image not found")
    
    # Filter valid images to prevent errors
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp',}
    r_imgs = [i for i in os.listdir(raw_img_dir) 
              if not os.path.isdir(os.path.join(raw_img_dir,i)) 
              and os.path.splitext(i)[1].lower() in valid_exts]
    
    # Process images 
    results = []
    for item in r_imgs:
        results.append(process_one_img(raw_img_dir, item))

    print(results)
    

#     with open(label_file, 'wt', encoding='utf-8') as lf:
#         for img_path, ocr_data, bill_obj in results:
#             if bill_obj:
#                 # Convert Pydantic model to dict, handling datetime serialization
#                 bill_dict = bill_obj.model_dump(mode='json')
#                 
#                 # Create a combined record
#                 record = {
#                     "image_path": img_path,
#                     "ocr_result": ocr_data,
#                     "label": bill_dict
#                 }
#                 
#                 # Write to jsonl
#                 lf.write(json.dumps(record, ensure_ascii=False) + "\n")
#             else:
#                 print(f"Skipping {img_path} due to processing error.")

if __name__ == "__main__":
    main()
