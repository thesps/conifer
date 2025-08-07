from pathlib import Path
import yaml

template = '''
pytest.{}:
    extends: .pytest
    image: {}
    variables:
        PYTESTFILE: {}
'''

images = {'build' : 'registry.cern.ch/ci4fpga/vivado:2024.1',
          'no build' : 'registry.cern.ch/ci4fpga/ubuntu'}

# check whether "build" method is called in the test -> needs different resources
def calls_build(test_filename):
    with open(test_filename) as f:
        content = f.read()
        return '.build(' in content
        
def generate_test_yaml(directory='.'):
    # List of test files to scan
    test_dir = Path(directory)
    test_files = list(test_dir.glob("test_*.py"))

    yml = None

    for test_file in test_files:
        file_name = str(test_file)
        name = file_name.replace('test_', '').replace('.py', '')
        build = 'build' if calls_build(test_file) else 'no build'
        test_yml = yaml.safe_load(template.format(name, images[build], test_file))
        if yml is None:
            yml = test_yml
        else:
            yml.update(test_yml)

    return yml

if __name__ == '__main__':
    yml = generate_test_yaml(Path(__file__).parent)
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile)