import easyocr

class ocr:
    def __init__(self, language = ['ch_tra','en']):
        self.reader = easyocr.Reader(language) # this needs to run only once to load the model into memory

    def __feature_grasp(self, raw_loc): # Here `b` stands for "boundary"
        left_b   = raw_loc[0][0]
        right_b  = raw_loc[2][0]
        upper_b  = raw_loc[0][1]
        lower_b  = raw_loc[2][1]
        x_center = (raw_loc[0][0] + raw_loc[2][0]) / 2
        y_center = (raw_loc[2][1] + raw_loc[0][1]) / 2
        height   = raw_loc[2][1] - raw_loc[0][1]
        data = {
            "left_b"   : left_b,
            "right_b"  : right_b,
            "upper_b"  : upper_b,
            "lower_b"  : lower_b,
            "x_center" : x_center,
            "y_center" : y_center,
            "height"   : height,
        }
        return data

    def infer(self, image):
        try:
            results = self.reader.readtext(image,) 
            for r_index in range(len(results)):
                results[r_index] = (self.__feature_grasp(results[r_index][0]), results[r_index][1])
        except Exception as e:
            print(f"Exception {e} occured")
            result = []
        return results
    
    def __y_sort(self, raw_result):
        raw_result.sort(key = lambda item: item[0]['upper_b'])
        return None

    def __is_overlap(self, a_box, b_box, ratio = 0.5): # In this situation, a/b_box is the whole box object, including loc, and str, etc. We would assume that for b, the upper_b is lower than that of a
        if a_box[0]['lower_b'] - b_box[0]['upper_b'] > b_box[0]['height'] * ratio:
            return True
        else:
            return False

    def __y_group(self, sorted_result):
        grouped_rows = []
        row_group    = [sorted_result[0]]
        for index in range(1, len(sorted_result)):
            if self.__is_overlap(row_group[-1],sorted_result[index]):
                row_group.append(sorted_result[index])
            else:
                grouped_rows.append(row_group)
                row_group = [sorted_result[index]]
                sorted_result[index]
        grouped_rows.append(row_group)
        return grouped_rows

    def __x_sort(self, row_group):
        row_group.sort(key = lambda element: element[0]['left_b'])
        return None

    def __li2str(self, g_result):
        strli = []
        for row in g_result:
            row_strli = []
            for element in row:
                row_strli.append(element[1])
            strli.append(' '.join(row_strli))
        return '\n'.join(strli)

    def after_care(self, raw_result):
        self.__y_sort(raw_result)
        g_result = self.__y_group(raw_result)
        for index in range(len(g_result)):
            self.__x_sort(g_result[index])
        
        return self.__li2str(g_result)




