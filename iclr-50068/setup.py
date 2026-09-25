from setuptools import find_packages, setup

setup(
    name="cycloformer",
    version="0.1.0",
    description="CycloFormer: rotation-invariant sEMG hand pose estimation",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "emg2pose": ["UmeTrack/dataset/generic_hand_model.json"],
    },
    install_requires=[],
)
