import easyocr

class ocr:
    def __init__(self, language = ['ch_tra','en']):
        self.reader = easyocr.Reader(language) # this needs to run only once to load the model into memory

    def infer(self, image, **kwargs):
        params = {
            'paragraph': True,
            'slope_ths': 0.1,
            'x_ths': 50.0,      
            'y_ths': 0.5,       
        }
        params.update(kwargs)
        try:
            result = self.reader.readtext(image, **params,)
        except Exception:
            print(f"Exception {e} occured")
            result = []
        return result

    def li2str(self, inferred_li):
        return '\n'.join(inferred_li)
        

