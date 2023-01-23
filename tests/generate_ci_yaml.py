import yaml
import glob

'''
Create a Gitlab CI yml file with a separate entry for each test_* file
in the pytests directory to parallelise the CI jobs.
'''

template = """
pytest.{}:
  extends: .pytest
  variables:
    PYTESTFILE: {}
"""

yml = None
tests = glob.glob('test_*.py')
for test in tests:
    name = test.replace('test_','').replace('.py','')
    new_yml = yaml.safe_load(template.format(name, 'test_{}.py'.format(name)))
    if yml is None:
        yml = new_yml
    else:
        yml.update(new_yml)

yamlfile = open('pytests.yml', 'w')
yaml.safe_dump(yml, yamlfile)