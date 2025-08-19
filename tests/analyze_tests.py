from pathlib import Path
import yaml

template = '''
pytest.{name}:
  extends: .pytest-{extends}
  variables:
    PYTESTFILE: {test_file}
  allow_failure: {allow_failure}
'''

# override the auto detection of which script to extend for the following jobs
extends_override = {'backends' : 'fpga'}

# allow the following jobs to fail
allow_failure = ['backends', 'xgb_converter', 'onnx_to_hls']

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
        build = calls_build(test_file)
        extends = 'fpga' if build else 'plain'
        if name in extends_override.keys():
            extends = extends_override[name]
        allow_fail = 'True' if name in allow_failure else 'False'
        test_yml = yaml.safe_load(template.format(name=name,
                                                  extends=extends,
                                                  test_file=test_file,
                                                  allow_failure=allow_fail))
        if yml is None:
            yml = test_yml
        else:
            yml.update(test_yml)

    return yml

if __name__ == '__main__':
    yml = generate_test_yaml(Path(__file__).parent)
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile)