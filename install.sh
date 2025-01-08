rm -rf build dist
python depth_gemm_C_setup.py install
rm -rf build dist
python setup.py bdist_wheel
pip install dist/*.whl