
from setuptools import find_packages , setup
from typing import List

HYPEHEN_E_DOT = "-e ."

def get_requirements(file_path : str) ->List[str]:
     """This return a list of requirements"""
     
     requirements = []
     
     with open(file_path) as file_obj :
         requirements = file_obj.readlines()
         requirements = [req.replace('\n', "") for req in requirements]
         
         if HYPEHEN_E_DOT  in requirements:
             requirements.remove(HYPEHEN_E_DOT)  
     return requirements

setup(
    name = 'STUDENT_ML_PROJECT',
    version= '0.0.1',
    author= 'Rotimi',
    author_email= "Rotimikolawole938@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt') # read ehatever library is in the requirements.file 
)
