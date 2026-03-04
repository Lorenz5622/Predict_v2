from setuptools import setup, find_packages
packages = find_packages()
setup(
    name="Qwen_MoE",
    version="0.1.0",
    packages=packages,                 # 自动发现所有包
)
