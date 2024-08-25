import sys
import traceback
import logging

def error_mssg_details(error):
    """Function to extract and format detailed error message with traceback"""
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: {file_name} at line {line_number} with message: {str(error)}"
    detailed_traceback = "".join(traceback.format_tb(exc_tb))
    return f"{error_message}\nTraceback:\n{detailed_traceback}"

class CustomException(Exception):
    """Base class for other exceptions"""
    def __init__(self, error, error_detail: sys):
        self.error_message = error_mssg_details(error)
        super().__init__(self.error_message)
    
    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":

    try :
        a = 10/0
    except Exception as e:
        
        logging.info("Division error")
        raise CustomException(e , sys)


