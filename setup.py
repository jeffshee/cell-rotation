import sys
from cx_Freeze import setup, Executable

include_files = ["config.json"]
build_exe_option = {"packages": ["seaborn"], "include_files": include_files}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="Cell Rotation",
    version="2.0",
    description="Python script to calculate the rotation speed of the cell from cell videos.",
    options={"build_exe": build_exe_option},
    executables=[Executable("gui_qt.py")]
)
