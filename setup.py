from setuptools import find_packages, setup
from typing import List

Hyper_E_Dot='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function returns list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        #need to remove the \n from the string being read from txt file 
        requirements=[req.replace("\n","") for req in requirements]
        
        if Hyper_E_Dot in requirements:
            requirements.remove(Hyper_E_Dot)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author ='john',
    author_mail='johntojoind@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)