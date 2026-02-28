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
from enum import StrEnum

dotenv.load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'

raw_img_dir = './raw_dataset'
label_file = './file.jsonl'

oc = ocr.ocr()

client = openai.OpenAI(api_key=api_key, base_url=base_url)

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
    amount:     str = pdt.Field(pattern=r'^\d+\.\d{2}$|^\d+$')
    category:   CategoryType

    @pdt.model_validator(mode='after')
    def validate_currency_precision(self):
        zero_decimal_currencies = {CurrencyType.JPY, CurrencyType.TWD}
        if self.currency in zero_decimal_currencies:
            if not re.match(r'^\d+$', self.amount):
                raise ValueError(f"{self.currency} amounts cannot have decimal places (received {self.amount}).")
        else:
            if not re.match(r'^\d+\.\d{2}$', self.amount):
                raise ValueError(f"{self.currency} only accepts amounts with 2 decimal places (received {self.amount}).")
        return self

# ---------------------------------------------------------------------------------------

system_prompt = \
"""
You are now a receipt reader, and the receipt will be provided in the form of an image by the user.
Try to extract the following information from the given image and output it as a JSON object:
- The Merchant
- Datetime (ISO 8601)
- Currency (ISO 4217)
- Total Amount Paid.
- Paid Currency
- Category (Enum in the schema)

Ensure the output is a valid JSON object matching this schema:
```
{
  "merchant": "string",
  "datetime": "YYYY-MM-DDTHH:MM:SS",
  "currency": "Enum(HKD, USD, JPY, CNY, TWD, EUR, GBP)",
  "amount": "string (formatted as '10.00' or '100')",
  "category": "Enum(Dining, Transportation, Groceries, Shopping, Services, Accommodation, Entertainment, Others)"
}
```

Do NOT include things like triple backtick or single backtick to wrap code. Simply return pure json string.
Notice, datetime might be in the English form, where it is displayed as dd/mm/yy.
"""

def get_base64_img(r_img_dir: str, r_img: str) -> tuple[str, str]:
    r_img_full_dir = os.path.join(r_img_dir, r_img)
    img_type = r_img.split('.')[-1]
    with open(r_img_full_dir, 'rb') as file:
        base64_img = base64.b64encode(file.read()).decode('utf-8')
    return base64_img, img_type

def get_gemini_response(system_prompt: str, img_type: str, base64_img: str) -> Bill | None:
    """
    Calls Gemini, parses the JSON response, and returns a validated Bill object.
    Returns None on any failure.
    """
    try:
        print('sending request...')
        response = client.chat.completions.create(
            model="gemini-3-flash-preview",  
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract data from this receipt image as JSON."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{img_type};base64,{base64_img}"}
                        },
                    ],
                },
            ],
        )
        print('request received')
        message = response.choices[0].message.content

        # FIX: parse as JSON dict and validate with Pydantic instead of returning a raw regex tuple
        parsed = json.loads(message)
        bill = Bill(**parsed)
        return bill

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        return None
    except pdt.ValidationError as e:
        print(f"Pydantic validation error: {e}")
        return None
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return None

def get_ocr_result(r_img_dir: str, r_img: str) -> str:
    return oc.after_care(oc.infer(os.path.join(r_img_dir, r_img)))  # FIX: removed trailing comma

def process_one_img(r_img_dir: str, r_img: str) -> tuple[str, str, Bill] | None:
    """
    Processes a single image through OCR and Gemini.
    Returns (img_path, ocr_result, Bill) on success, or None on failure.
    """
    img_path = os.path.join(r_img_dir, r_img)

    print("calculating b64 image...")
    b64img, img_type = get_base64_img(r_img_dir, r_img)

    print("obtaining OCR result...")
    ocr_result = get_ocr_result(r_img_dir, r_img)

    print("obtaining VLM result...")
    bill = get_gemini_response(system_prompt, img_type, b64img)

    print("=" * 74)
    print(img_path)
    print("-" * 74)
    print(ocr_result)
    print("-" * 74)
    print(bill)
    print("=" * 74)

    # FIX: return data when bill is valid; return None when it isn't
    if bill:
        return img_path, ocr_result, bill
    else:
        print(f"None recorded for {img_path}")
        return None

def main():
    if not os.path.exists(raw_img_dir):
        raise ValueError("target directory for raw image not found")

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    r_imgs = [
        i for i in os.listdir(raw_img_dir)
        if not os.path.isdir(os.path.join(raw_img_dir, i))
        and os.path.splitext(i)[1].lower() in valid_exts
    ]

    results = []
    for item in r_imgs:
        one_image = process_one_img(raw_img_dir, item)
        # FIX: append when result is valid (not None), discard failures
        if one_image:
            results.append(one_image)

    print("Results length:", len(results))

    with open(label_file, 'at', encoding='utf-8') as lf:
        for img_path, ocr_data, bill_obj in results:
            bill_dict = bill_obj.model_dump(mode='json')
            record = {
                "image_path": img_path,
                "ocr_result": ocr_data,
                "label": bill_dict
            }
            lf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. {len(results)} records written to {label_file}.")

if __name__ == "__main__":
    main()
