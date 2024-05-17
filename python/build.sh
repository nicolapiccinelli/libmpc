# check if we want to build the package or build inplace using a flag
if [ "$1" == "package" ]; then
    echo "Building package"
    python3 setup.py sdist bdist_wheel
else
    echo "Building inplace"
    python3 setup.py build_ext --inplace
fi