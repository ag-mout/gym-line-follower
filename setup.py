# from setuptools import setup
import setuptools

setuptools.setup(name='gym_line_follower',
      version='0.1.2',
      install_requires=['gym',
                        'pybullet', 'opencv-python', 'shapely', 'numpy'],
      author="Nejc Planinsek",
      author_email="planinseknejc@gmail.com",
      description="Line follower simulator environment.",
      packages = setuptools.find_packages()
      )
