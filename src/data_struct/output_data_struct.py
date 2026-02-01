import pydantic as pdt
from pydantic import BaseModel

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
import re


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
    amount:     str = pdt.Field(pattern = r'^\d+\.\d{2}$ | ^\d+$') 
    category:   CategoryType

    @pdt.model_validator(mode='after')
    def validate_currency_precision(self):
        zero_decimal_currencies = {CurrencyType.JPY, CurrencyType.TWD}
        if self.currency in zero_decimal_currencies:
            if not re.match(r'^\d+$', self.amount): 
                raise ValueError(f"{self.currency} amounts cannot have decimal places (received {self.amount}).")
        else:
            if not re.match(r'^\d+\.\d{2}$'):
                raise ValueError(f"{self.currency} only accept amount with 2 decimal places (received {self.amount}).")
        return self
