from setuptools import setup, find_packages
from glob import glob

name = 'autolife_shelf_grasp_306'
setup(name=name, version='0.1.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages', ['resource/'+name]),
                  ('share/'+name, ['package.xml','README_ZH.md']),
                  ('share/'+name+'/config', glob('config/*.yaml')),
                  ('share/'+name+'/launch', glob('launch/*.py'))],
      install_requires=['setuptools'], zip_safe=False,
      entry_points={'console_scripts':['shelf_grasp = autolife_shelf_grasp_306.app:main']})
