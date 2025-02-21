from tests import util
import numpy as np
import conifer
import json
import os

'''
Test conifer's model saving and loading functionality by loading some models and checking the predictions
match the original.
'''
def test_hls_save_load(hls_convert, train_skl):
  orig_model = hls_convert
  clf, X, y = train_skl
  load_model = conifer.model.load_model(f'{orig_model.config.output_dir}/{orig_model.config.project_name}.json')
  load_model.config.output_dir += '_loaded'
  load_model.compile()
  y_hls_0, y_hls_1 = util.predict_skl(orig_model, X, y, load_model)
  np.testing.assert_array_equal(y_hls_0, y_hls_1)

def test_hls_reload_last_shared_library(hls_convert, train_skl):
  clf, X, y = train_skl
  initial_model = conifer.model.load_model(f'{hls_convert.config.output_dir}/{hls_convert.config.project_name}.json', shared_library = False)
  initial_model.config.output_dir += '_loaded'
  initial_model.compile()
  # Re-load without recompiling to check if the shared library is loaded correctly
  reload_model = conifer.model.load_model(f'{hls_convert.config.output_dir}_loaded/{hls_convert.config.project_name}.json', shared_library=True)
  y_hls, y_hls_reload = util.predict_skl(initial_model, X, y, reload_model)
  np.testing.assert_array_equal(y_hls, y_hls_reload)
  assert os.path.abspath(initial_model.bridge.__file__) == os.path.abspath(reload_model.bridge.__file__), "Loaded two different shared libraries"

def test_hls_reload_manual_shared_library(hls_convert, train_skl):
  clf, X, y = train_skl
  initial_model = conifer.model.load_model(f'{hls_convert.config.output_dir}/{hls_convert.config.project_name}.json', shared_library = False)
  initial_model.config.output_dir += '_loaded'
  initial_model.compile()
  so_path = os.path.abspath(initial_model.bridge.__file__) # manually get the shared library path
  # Re-load without recompiling to check if the shared library is loaded correctly
  reload_model = conifer.model.load_model(f'{hls_convert.config.output_dir}_loaded/{hls_convert.config.project_name}.json', shared_library=so_path) # pass the shared library path manually
  y_hls, y_hls_reload = util.predict_skl(initial_model, X, y, reload_model)
  np.testing.assert_array_equal(y_hls, y_hls_reload)
  assert os.path.abspath(initial_model.bridge.__file__) == os.path.abspath(reload_model.bridge.__file__), "Loaded two different shared libraries"

def test_hdl_save_load(vhdl_convert, train_skl):
  orig_model = vhdl_convert
  clf, X, y = train_skl
  load_model = conifer.model.load_model(f'{orig_model.config.output_dir}/{orig_model.config.project_name}.json')
  load_model.config.output_dir += '_loaded'
  load_model.compile()
  y_hdl_0, y_hdl_1 = util.predict_skl(orig_model, X, y, load_model)
  np.testing.assert_array_equal(y_hdl_0, y_hdl_1)

def test_new_config(hls_convert, train_skl):
  orig_model = hls_convert
  clf, X, y = train_skl
  with open(f'{orig_model.config.output_dir}/{orig_model.config.project_name}.json') as json_file:
    js = json.load(json_file)
  cfg = js['config']
  cfg['backend'] = 'cpp'
  cfg['output_dir'] += '_loaded'
  load_model = conifer.model.load_model(f'{orig_model.config.output_dir}/{orig_model.config.project_name}.json', new_config=cfg)
  load_model.compile()
  y_hls_0, y_cpp_1 = util.predict_skl(orig_model, X, y, load_model)
  assert load_model.config.backend == 'cpp', "Model backend was not successfuly modified"
  np.testing.assert_array_equal(y_hls_0, y_cpp_1.reshape(y_hls_0.shape))