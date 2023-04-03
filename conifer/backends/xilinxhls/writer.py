import os
from shutil import copyfile
import warnings
import numpy as np
import copy
from conifer.utils import _ap_include, copydocstring
from conifer.backends.common import BottomUpDecisionTree, MultiPrecisionConfig, read_hls_report
from conifer.model import ModelBase
import datetime
import logging
logger = logging.getLogger(__name__)

_TOOLS = {
    'vivadohls': 'vivado_hls',
    'vitishls': 'vitis_hls'
}


def get_tool_exe_in_path(tool):
    if tool not in _TOOLS.keys():
        return None

    tool_exe = _TOOLS[tool]

    if os.system('which {} > /dev/null 2>/dev/null'.format(tool_exe)) != 0:
        return None

    return tool_exe


def get_hls():

    tool_exe = None

    if '_tool' in globals():
        tool_exe = get_tool_exe_in_path(_tool)
    else:
        for tool in _TOOLS.keys():
            tool_exe = get_tool_exe_in_path(tool)
            if tool_exe != None:
                break

    return tool_exe

class XilinxHLSConfig(MultiPrecisionConfig):
    backend = 'xilinxhls'
    _config_fields = MultiPrecisionConfig._config_fields + ['xilinx_part', 'clock_period', 'unroll']
    _xhls_alts = {'xilinx_part'  : ['XilinxPart'],
                  'clock_period' : ['ClockPeriod'],
                  'unroll'       : ['Unroll']
                  }
    _alternates = {**MultiPrecisionConfig._alternates, **_xhls_alts}
    _xhls_defaults = {'precision'    : 'ap_fixed<18,8>',
                      'xilinx_part'  : 'xcvu9p-flgb2104-2L-e',
                      'clock_period' : 5,
                      'unroll'     : True
                      }
    _defaults = {**MultiPrecisionConfig._defaults, **_xhls_defaults}
    def __init__(self, configDict, validate=True):
        super(XilinxHLSConfig, self).__init__(configDict, validate=False)
        if validate:
            self._validate()

    def default_config():
        return copy.deepcopy(XilinxHLSConfig._defaults)

class XilinxHLSModel(ModelBase):

    def __init__(self, ensembleDict, config, metadata=None):
        super(XilinxHLSModel, self).__init__(ensembleDict, config, metadata)
        self.config = XilinxHLSConfig(config)
        trees = ensembleDict.get('trees', None)
        assert trees is not None, f'Missing expected key trees in ensembleDict'
        self.trees = [[BottomUpDecisionTree(treeDict) for treeDict in trees_class] for trees_class in trees]
        for trees_class in self.trees:
            for tree in trees_class:
                tree.padTree(self.max_depth)

    def write_bdt_h(self):
        '''
        Write the BDT.h file depending on configuration options
        '''
        cfg = self.config
        filedir = os.path.dirname(os.path.abspath(__file__))
        
        if cfg.unroll:
            copyfile(f'{filedir}/firmware/BDT_unrolled.h',
                     f'{cfg.output_dir}/firmware/BDT.h')    
        else:        
            copyfile(f'{filedir}/firmware/BDT_rolled.h',
                     f'{cfg.output_dir}/firmware/BDT.h')

        if cfg.unroll:
            fin = open(f'{filedir}/hls-template/firmware/BDT_unrolled.cpp', 'r')
            fout = open(f'{cfg.output_dir}/firmware/BDT.cpp', 'w')
            for line in fin.readlines():
                if '// conifer insert tree_scores' in line:
                    newline = ''
                    for it, trees in enumerate(self.trees):
                        for ic, tree in enumerate(trees):
                            newline += f'  scores[{it}][{ic}] = tree_{it}_{ic}.decision_function(x);\n'
                else:
                    newline = line
                fout.write(newline)
        else:
            copyfile(f'{filedir}/hls-template/firmware/BDT_rolled.cpp',
                     f'{cfg.output_dir}/firmware/BDT.cpp')


    def write_parameters_h(self):
        '''
        Write the parameters.h file depending on the configuration
        '''

        cfg = self.config
        
        fout = open('{}/firmware/parameters.h'.format(cfg.output_dir), 'w')
        fout.write('#ifndef BDT_PARAMS_H__\n#define BDT_PARAMS_H__\n\n')
        fout.write('#include  "BDT.h"\n')
        fout.write('#include "ap_fixed.h"\n\n')
        fout.write('static const int n_trees = {};\n'.format(
            self.n_trees))
        fout.write('static const int max_depth = {};\n'.format(
            self.max_depth))
        fout.write('static const int n_features = {};\n'.format(
            self.n_features))
        fout.write('static const int n_classes = {};\n'.format(
            self.n_classes))
        fout.write('static const bool unroll = {};\n'.format(
            str(cfg.unroll).lower()))

        input_precision = cfg.input_precision
        fout.write('typedef {} input_t;\n'.format(input_precision))
        fout.write('typedef input_t input_arr_t[n_features];\n')

        threshold_precision = cfg.threshold_precision
        fout.write('typedef {} threshold_t;\n'.format(threshold_precision))

        score_precision = cfg.score_precision
        fout.write('typedef {} score_t;\n'.format(score_precision))
        fout.write('typedef score_t score_arr_t[n_classes];\n')

        if self.config.unroll:
            self._write_parameters_h_unrolled(fout)
        else:
            self._write_parameters_h_array(fout)
        fout.close()       

    def _write_parameters_h_unrolled(self, fout):
        '''
        Write the parameters.h file with manually unrolled trees
        '''
        cfg = self.config
        tree_fields = ['feature', 'threshold', 'value',
                       'children_left', 'children_right', 'parent']

        # write the BDT instance
        fout.write(
            "static const BDT::BDT<n_trees, n_classes, input_arr_t, score_t, threshold_t> bdt = \n")
        fout.write("{ // The struct\n")
        newline = "\t" + str(self.norm) + ", // The normalisation\n"
        fout.write(newline)
        newline = "\t{"
        if self.n_classes > 2:
            for iip, ip in enumerate(self.init_predict):
                if iip < len(self.init_predict) - 1:
                    newline += '{},'.format(ip)
                else:
                    newline += '{}}}, // The init_predict\n}};'.format(ip)
        else:
            newline += str(self.init_predict[0]) + '},\n}; // bdt\n'
        fout.write(newline)

        # write the trees instances
        nc = 1 if self.n_classes == 2 else self.n_classes
        fout.write("// The trees\n")
        # loop over trees
        for itree, trees in enumerate(self.trees):
            # loop over classes
            for iclass, tree in enumerate(trees):
                fout.write(f'static const BDT::Tree<{itree*nc+iclass}, {tree.n_nodes()}, {tree.n_leaves()}')
                fout.write(f', input_arr_t, score_t, threshold_t>')
                fout.write(f' tree_{itree}_{iclass} = {{\n')
                # loop over fields
                for ifield, field in enumerate(tree_fields):
                    newline = '    {'
                    newline += ','.join(map(str, getattr(tree, field)))
                    newline += '}'
                    if ifield < len(tree_fields) - 1:
                        newline += ','
                    newline += '\n'
                    fout.write(newline)
                fout.write('};\n')
            #fout.write(newline)
        fout.write('#endif')

    def _write_parameters_h_array(self, fout):
        '''
        Write the parameters.h file with the array of trees
        '''

        tree_fields = ['feature', 'threshold', 'value',
                    'children_left', 'children_right', 'parent']

        fout.write(
            "static const BDT::BDT<n_trees, max_depth, n_classes, input_arr_t, score_t, threshold_t, unroll> bdt = \n")
        fout.write("{ // The struct\n")
        newline = "\t" + str(self.norm) + ", // The normalisation\n"
        fout.write(newline)
        newline = "\t{"
        if self.n_classes > 2:
            for iip, ip in enumerate(self.init_predict):
                if iip < len(self.init_predict) - 1:
                    newline += '{},'.format(ip)
                else:
                    newline += '{}}}, // The init_predict\n'.format(ip)
        else:
            newline += str(self.init_predict[0]) + '},\n'
        fout.write(newline)
        fout.write("\t{ // The array of trees\n")
        # loop over trees
        for itree, trees in enumerate(self.trees):
            fout.write('\t\t{ // trees[' + str(itree) + ']\n')
            # loop over classes
            for iclass, tree in enumerate(trees):
                fout.write('\t\t\t{ // [' + str(iclass) + ']\n')
                # loop over fields
                for ifield, field in enumerate(tree_fields):
                    newline = '\t\t\t\t{'
                    newline += ','.join(map(str, getattr(tree, field)))
                    newline += '}'
                    if ifield < len(tree_fields) - 1:
                        newline += ','
                    newline += '\n'
                    fout.write(newline)
                newline = '\t\t\t}'
                if iclass < len(trees) - 1:
                    newline += ','
                newline += '\n'
                fout.write(newline)
            newline = '\t\t}'
            if itree < self.n_trees - 1:
                newline += ','
            newline += '\n'
            fout.write(newline)
        fout.write('\t}\n};')
        fout.write('\n#endif')

    @copydocstring(ModelBase.write)
    def write(self):

        self.save()
        cfg = self.config

        filedir = os.path.dirname(os.path.abspath(__file__))

        logger.info(f"Writing project to {cfg.output_dir}")

        os.makedirs('{}/firmware'.format(cfg.output_dir), exist_ok=True)
        os.makedirs('{}/tb_data'.format(cfg.output_dir), exist_ok=True)

        self.write_bdt_h()
        self.write_parameters_h()

        ###################
        # myproject.cpp
        ###################

        fout = open(
            '{}/firmware/{}.cpp'.format(cfg.output_dir, cfg.project_name), 'w')
        fout.write('#include "BDT.h"\n')
        fout.write('#include "parameters.h"\n')
        fout.write('#include "{}.h"\n'.format(cfg.project_name))

        fout.write(
            'void {}(input_arr_t x, score_arr_t score){{\n'.format(cfg.project_name))
        fout.write('\t#pragma HLS array_partition variable=x\n')
        fout.write('\t#pragma HLS array_partition variable=score\n')
        if(cfg.unroll):
            fout.write('\t#pragma HLS pipeline\n')
            fout.write('\t#pragma HLS unroll\n')
        fout.write('\tbdt.decision_function(x, score);\n}')
        fout.close()

        #######################
        # myproject.h
        #######################

        f = open(os.path.join(filedir, 'hls-template/firmware/myproject.h'), 'r')
        fout = open(
            '{}/firmware/{}.h'.format(cfg.output_dir, cfg.project_name), 'w')

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace(
                    'MYPROJECT', format(cfg.project_name.upper()))
            elif 'void myproject(' in line:
                newline = 'void {}(\n'.format(cfg.project_name)
            elif 'hls-fpga-machine-learning insert args' in line:
                newline = '\tinput_arr_t data,\n\tscore_arr_t score,\n\tscore_t tree_scores[BDT::fn_classes(n_classes) * n_trees]);'
            # Remove some lines

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

        #######################
        # myproject_test.cpp
        #######################

        f = open(os.path.join(filedir, 'hls-template/myproject_test.cpp'))
        fout = open(
            '{}/{}_test.cpp'.format(cfg.output_dir, cfg.project_name), 'w')

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            # Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', cfg.project_name)
            elif '//hls-fpga-machine-learning insert data' in line:
                newline = line
                newline += '      std::vector<double>::const_iterator in_begin = in.cbegin();\n'
                newline += '      std::vector<double>::const_iterator in_end;\n'
                newline += '      input_arr_t x;\n'
                newline += '      in_end = in_begin + ({});\n'.format(
                    self.n_features)
                newline += '      std::copy(in_begin, in_end, x);\n'
                newline += '      in_begin = in_end;\n'
                # brace-init zeros the array out because we use std=c++0x
                newline += '      score_arr_t score{};\n'
                newline += '      score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]{};\n'
                # but we can still explicitly zero out if you want
                newline += '      std::fill_n(score, {}, 0.);\n'.format(
                    self.n_classes)
            elif '//hls-fpga-machine-learning insert zero' in line:
                newline = line
                newline += '    input_arr_t x;\n'
                newline += '    std::fill_n(x, {}, 0.);\n'.format(
                    self.n_features)
                newline += '    score_arr_t score{};\n'
                newline += '    score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]{};\n'
                newline += '    std::fill_n(score, {}, 0.);\n'.format(
                    self.n_classes)
            elif '//hls-fpga-machine-learning insert top-level-function' in line:
                newline = line
                top_level = indent + \
                    '{}(x, score, tree_scores);\n'.format(cfg.project_name)
                newline += top_level
            elif '//hls-fpga-machine-learning insert predictions' in line:
                newline = line
                newline += indent + \
                    'for(int i = 0; i < {}; i++) {{\n'.format(
                        self.n_classes)
                newline += indent + '  std::cout << pr[i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'std::cout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                newline += indent + \
                    'for(int i = 0; i < {}; i++) {{\n'.format(
                        self.n_classes)
                newline += indent + '  fout << score[i] << " ";\n'
                newline += indent + '}\n'
            elif '//hls-fpga-machine-learning insert output' in line or '//hls-fpga-machine-learning insert quantized' in line:
                newline = line
                newline += indent + \
                    'for(int i = 0; i < {}; i++) {{\n'.format(
                        self.n_classes)
                newline += indent + '  std::cout << score[i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'std::cout << std::endl;\n'
            else:
                newline = line
            fout.write(newline)

        fout.close()

        #######################
        # build_prj.tcl
        #######################

        bdtdir = os.path.abspath(os.path.join(filedir, "../bdt_utils"))
        relpath = os.path.relpath(bdtdir, start=cfg.output_dir)

        f = open(os.path.join(filedir, 'hls-template/build_prj.tcl'), 'r')
        fout = open('{}/build_prj.tcl'.format(cfg.output_dir), 'w')

        for line in f.readlines():

            line = line.replace('nnet_utils', relpath)
            line = line.replace('myproject', cfg.project_name)

            if 'set_part {xc7vx690tffg1927-2}' in line:
                line = 'set_part {{{}}}\n'.format(cfg.xilinx_part)
            elif 'create_clock -period 5 -name default' in line:
                line = 'create_clock -period {} -name default\n'.format(
                    cfg.clock_period)
            # Remove some lines
            elif ('weights' in line) or ('-tb firmware/weights' in line):
                line = ''

            fout.write(line)
        f.close()
        fout.close()

        #######################
        # bridge.cpp
        #######################

        copyfile(f'{filedir}/hls-template/bridge.cpp',
            f"{cfg.output_dir}/bridge_tmp.cpp")

        fin = open(f"{cfg.output_dir}/bridge_tmp.cpp", 'r')
        fout = open(f"{cfg.output_dir}/bridge.cpp", 'w')
        for line in fin.readlines():
            newline = line
            if 'PYBIND11_MODULE' in line:
                newline = f'PYBIND11_MODULE(conifer_bridge_{self._stamp}, m){{\n'
            fout.write(newline)
        fin.close()
        fout.close()
        os.remove(f"{cfg.output_dir}/bridge_tmp.cpp")

    @copydocstring(ModelBase.decision_function)
    def decision_function(self, X, trees=False):
        cfg = self.config
        curr_dir = os.getcwd()
        os.chdir(cfg.output_dir)
        if len(X.shape) == 1:
            y = np.array(self.bridge.decision_function(X))
        elif len(X.shape) == 2:
            y = np.array([self.bridge.decision_function(xi) for xi in X])
        else:
            raise Exception(f"Can't handle data shape {X.shape}, expected 1D or 2D shape")
        os.chdir(curr_dir)
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(y.shape[0])
        return y
    
    @copydocstring(ModelBase.compile)
    def compile(self):
        self.write()
        cfg = self.config
        curr_dir = os.getcwd()
        os.chdir(cfg.output_dir)
        ap_include = _ap_include()
        if ap_include is None:
            os.chdir(curr_dir)
            raise Exception("Couldn't find Xilinx ap_ headers. Source the Vivado/Vitis HLS toolchain, or set XILINX_AP_INCLUDE environment variable.")
        cmd = f"g++ -O3 -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) {ap_include} bridge.cpp firmware/BDT.cpp firmware/{cfg.project_name}.cpp -o conifer_bridge_{self._stamp}.so"
        logger.debug(f'Compiling with command {cmd}')
        try:
            ret_val = os.system(cmd)
            if ret_val != 0:
                raise Exception(f'Failed to compile project {cfg.project_name}')
        except:
            os.chdir(curr_dir)
            raise Exception(f'Failed to compile project {cfg.project_name}')

        try:
            logger.debug(f'Importing conifer_bridge_{self._stamp} from conifer_bridge_{self._stamp}.so')
            import importlib.util
            spec = importlib.util.spec_from_file_location(f'conifer_bridge_{self._stamp}', f'./conifer_bridge_{self._stamp}.so')
            self.bridge = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.bridge)
        except ImportError:
            os.chdir(curr_dir)
            raise Exception("Can't import pybind11 bridge, is it compiled?")
        finally:
            os.chdir(curr_dir)

    @copydocstring(ModelBase.build)
    def build(self, reset=False, csim=False, synth=True, cosim=False, export=False):
        cwd = os.getcwd()
        os.chdir(self.config.output_dir)
        
        rval = True
        hls_tool = get_hls()
        if hls_tool is None:
            logger.error("No HLS in PATH. Did you source the appropriate Xilinx Toolchain?")
            rval = False
        else:
            cmd = '{hls_tool} -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} export={export}" > build.log'\
                .format(hls_tool=hls_tool, reset=reset, csim=csim, synth=synth, cosim=cosim, export=export)
            start = datetime.datetime.now()
            logger.info(f'build starting {start:%H:%M:%S}')
            logger.debug(f'build invoking {hls_tool} with command "{cmd}"')
            success = os.system(cmd)
            stop = datetime.datetime.now()
            logger.info(f'build finished {stop:%H:%M:%S} - took {str(stop-start)}')
            if(success > 0):
                logger.error("build failed, check logs")
                rval = False
        os.chdir(cwd)
        return rval

    def read_report(self) -> dict:
        '''
        Read the HLS C Synthesis report
        Returns
        ----------
        dictionary of extracted report contents
        '''
        report_file = f'{self.config.output_dir}/{self.config.project_name}_prj/solution1/syn/report/{self.config.project_name}_csynth.xml'
        full_report = read_hls_report(report_file)
        lb, lw = full_report['latency_best'], full_report['latency_worst']
        if lb != lw:
            logger.warn(f'Model has different best/worst latency ({lb} and {lw})')
        iib, iiw = full_report['interval_best'], full_report['interval_worst']
        if lb != lw:
            logger.warn(f'Model has different best/worst interval ({iib} and {iiw})')

        report = {}
        report['latency'] = lb
        report['interval'] = iib
        for key in ['lut', 'ff']:
            report[key] = full_report[key]
        return report

def auto_config(granularity='simple'):
    '''
    Create an initial configuration dictionary to modify
    Parameters
    ----------
    granularity : string, optional
        Which granularity to fill the template. Can be 'simple' (default) or 'full'
        If 'simple', only 'Precision' is included. If 'full', 'InputPrecision', 'ThresholdPrecision', and 'ScorePrecision'
        are included.
    '''
    config = {'Backend' : 'xilinxhls',
              'ProjectName': 'my_prj',
              'OutputDir': 'my-conifer-prj',
              'XilinxPart': 'xcvu9p-flgb2104-2L-e',
              'ClockPeriod': '5',
              'Unroll' : True}
    if granularity == 'full':
        config['InputPrecision'] = 'ap_fixed<18,8>'
        config['ThresholdPrecision'] = 'ap_fixed<18,8>'
        config['ScorePrecision'] = 'ap_fixed<18,8>'
    else:
        config['Precision'] = 'ap_fixed<18,8>'

    return config

def make_model(ensembleDict, config):
    return XilinxHLSModel(ensembleDict, config)
