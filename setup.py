import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(os.path.dirname(__file__), requirements_file),
              'r') as f:
        requirements = f.read().splitlines()
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


def get_version():
    """ Find the version of the package"""
    version_file = os.path.join(BASEDIR, 'chatterbox_onnx', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if alpha and int(alpha) > 0:
        version += f"a{alpha}"
    return version


setup(
    name='chatterbox_onnx',
    version=get_version(),
    packages=['chatterbox_onnx'],
    include_package_data=True,
    install_requires=required('requirements.txt'),
    url='https://github.com/TigreGotico/chatterbox_onnx',
    license='MIT',
    author='JarbasAi',
    author_email='jarbasai@mailfence.com',
    description='',
    entry_points={
        'mycroft.plugin.tts': 'ovos-tts-plugin-chatterbox-onnx = chatterbox_onnx.opm:ChatterboxTTSPlugin',
        'opm.transformer.tts': 'ovos-tts-transformer-chatterbox-onnx=chatterbox_onnx.opm:ChatterboxTTSTransformer',
        'console_scripts': [
            'chatterbox_bulk_vc=chatterbox_onnx.scripts:bulk_vc',
            'chatterbox_bulk_tts=chatterbox_onnx.scripts:bulk_tts'
        ]
    }
)
