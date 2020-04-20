import os
import sys
from shutil import copyfile
import numpy as np

def write(ensemble_dict, cfg):

    filedir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs('{}/firmware'.format(cfg['OutputDir']))
    os.makedirs('{}/tb_data'.format(cfg['OutputDir']))
    copyfile('{}/firmware/BDT.h'.format(filedir), '{}/firmware/BDT.h'.format(cfg['OutputDir']))

    ###################
    ## myproject.cpp
    ###################

    fout = open('{}/firmware/{}.cpp'.format(cfg['OutputDir'], cfg['ProjectName']),'w')
    fout.write('#include "BDT.h"\n')
    fout.write('#include "parameters.h"\n')
    fout.write('#include "{}.h"\n'.format(cfg['ProjectName']))

    fout.write('void {}(input_arr_t x, score_arr_t score){{\n'.format(cfg['ProjectName']))
    # TODO: probably only one of the pragmas is necessary?
    #fout.write('\t#pragma HLS pipeline II = {}\n'.format(cfg['ReuseFactor']))
    #fout.write('\t#pragma HLS unroll factor = {}\n'.format(cfg['ReuseFactor']))
    fout.write('\t#pragma HLS array_partition variable=x\n\n')
    fout.write('\t#pragma HLS array_partition variable=score\n\n')
    fout.write('\tbdt.decision_function(x, score);\n}')
    fout.close()

    ###################
    ## parameters.h
    ###################

    #f = open(os.path.join(filedir,'../hls-template/firmware/parameters.h'),'r')
    fout = open('{}/firmware/parameters.h'.format(cfg['OutputDir']),'w')
    fout.write('#ifndef BDT_PARAMS_H__\n#define BDT_PARAMS_H__\n\n')
    fout.write('#include  "BDT.h"\n')
    fout.write('#include "ap_fixed.h"\n\n')
    fout.write('static const int n_trees = {};\n'.format(ensemble_dict['n_trees']))
    fout.write('static const int max_depth = {};\n'.format(ensemble_dict['max_depth']))
    fout.write('static const int n_features = {};\n'.format(ensemble_dict['n_features']))
    fout.write('static const int n_classes = {};\n'.format(ensemble_dict['n_classes']))
    fout.write('typedef {} input_t;\n'.format(cfg['Precision']))
    fout.write('typedef input_t input_arr_t[n_features];\n')
    fout.write('typedef {} score_t;\n'.format(cfg['Precision']))
    fout.write('typedef score_t score_arr_t[n_classes];\n')
    # TODO score_arr_t
    fout.write('typedef input_t threshold_t;\n\n')

    tree_fields = ['feature', 'threshold', 'value', 'children_left', 'children_right', 'parent']

    fout.write("static const BDT::BDT<n_trees, max_depth, n_classes, input_arr_t, score_t, threshold_t> bdt = \n")
    fout.write("{ // The struct\n")
    newline = "\t" + str(ensemble_dict['norm']) + ", // The normalisation\n"
    fout.write(newline)
    newline = "\t{"
    if ensemble_dict['n_classes'] > 2:
        for iip, ip in enumerate(ensemble_dict['init_predict']):
            if iip < len(ensemble_dict['init_predict']) - 1:
                newline += '{},'.format(ip)
            else:
                newline += '{}}, // The init_predict\n'.format(ip)
    else:
        newline += str(ensemble_dict['init_predict']) + '},\n'
    fout.write(newline)
    fout.write("\t{ // The array of trees\n")
    # loop over trees
    for itree, trees in enumerate(ensemble_dict['trees']):
        fout.write('\t\t{ // trees[' + str(itree) + ']\n')
        # loop over classes
        for iclass, tree in enumerate(trees):
            fout.write('\t\t\t{ // [' + str(iclass) + ']\n')
            # loop over fields
            for ifield, field in enumerate(tree_fields):
                newline = '\t\t\t\t{'
                newline += ','.join(map(str, tree[field]))
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
        if itree < ensemble_dict['n_trees'] - 1:
          newline += ','
        newline += '\n'
        fout.write(newline)
    fout.write('\t}\n};')

    fout.write('\n#endif')
    fout.close()

    #######################
    ## myproject.h
    #######################

    f = open(os.path.join(filedir,'hls-template/firmware/myproject.h'),'r')
    fout = open('{}/firmware/{}.h'.format(cfg['OutputDir'], cfg['ProjectName']),'w')

    for line in f.readlines():

        if 'MYPROJECT' in line:
            newline = line.replace('MYPROJECT',format(cfg['ProjectName'].upper()))
        elif 'void myproject(' in line:
            newline = 'void {}(\n'.format(cfg['ProjectName'])
        elif 'input_t data[N_INPUTS]' in line:
            newline = '\tinput_arr_t data,\n\tscore_arr_t score);'
        # Remove some lines
        elif ('result_t' in line) or ('unsigned short' in line):
            newline = ''
        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()

    #######################
    ## myproject_test.cpp
    #######################

    f = open(os.path.join(filedir, 'hls-template/myproject_test2.cpp'))
    fout = open('{}/{}_test.cpp'.format(cfg['OutputDir'], cfg['ProjectName']),'w')

    for line in f.readlines():
        indent = ' ' * (len(line) - len(line.lstrip(' ')))

        #Insert numbers
        if 'myproject' in line:
            newline = line.replace('myproject', cfg['ProjectName'])
        elif '//hls-fpga-machine-learning insert data' in line:
            newline = line
            newline += '      std::vector<float>::const_iterator in_begin = in.cbegin();\n'
            newline += '      std::vector<float>::const_iterator in_end;\n'
            newline += '      input_arr_t x;\n'
            newline += '      in_end = in_begin + ({});\n'.format(ensemble_dict['n_features'])
            newline += '      std::copy(in_begin, in_end, x);\n'
            newline += '      in_begin = in_end;\n'
            # brace-init zeros the array out because we use std=c++0x
            newline += '      score_arr_t score{};\n'
            # but we can still explicitly zero out if you want
            newline += '      std::fill_n(score, {}, 0.);\n'.format(ensemble_dict['n_classes'])
        elif '//hls-fpga-machine-learning insert zero' in line:
            newline = line
            newline += '    input_arr_t x;\n'
            newline += '    std::fill_n(x, {}, 0.);\n'.format(ensemble_dict['n_features'])
            newline += '    score_arr_t score{};\n'
            newline += '      std::fill_n(score, {}, 0.);\n'.format(ensemble_dict['n_classes'])
        elif '//hls-fpga-machine-learning insert top-level-function' in line:
            newline = line
            top_level = indent + '{}(x, score);\n'.format(cfg['ProjectName'])
            newline += top_level
        elif '//hls-fpga-machine-learning insert predictions' in line:
            newline = line
            newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(ensemble_dict['n_classes'])
            newline += indent + '  std::cout << pr[i] << " ";\n'
            newline += indent + '}\n'
            newline += indent + 'std::cout << std::endl;\n'
        elif '//hls-fpga-machine-learning insert tb-output' in line:
            newline = line
            newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(ensemble_dict['n_classes'])
            newline += indent + '  fout << score[i] << " ";\n'
            newline += indent + '}\n'
            newline += indent + 'fout << std::endl;\n'
        elif '//hls-fpga-machine-learning insert output' in line or '//hls-fpga-machine-learning insert quantized' in line:
            newline = line
            newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(ensemble_dict['n_classes'])
            newline += indent + '  std::cout << score[i] << " ";\n'
            newline += indent + '}\n'
            newline += indent + 'std::cout << std::endl;\n'
        else:
            newline = line
        fout.write(newline)
    #fout.write('#include "BDT.h"\n')
    #fout.write('#include "firmware/parameters.h"\n')
    #fout.write('#include "firmware/{}.h"\n'.format(cfg['ProjectName']))

    #fout.write('int main(){\n')
    #fout.write('\tinput_arr_t x = {{{}}};\n'.format(str([0] * ensemble_dict['n_features'])[1:-1]));
    #fout.write('\tscore_arr_t score;\n')
    #fout.write('\t{}(x, score);\n'.format(cfg['ProjectName']))
    #fout.write('\tfor(int i = 0; i < n_classes; i++){\n')
    #fout.write('\t\tstd::cout << score[i] << ", ";\n\t}\n')
    #fout.write('\tstd::cout << std::endl;\n')
    #fout.write('\treturn 0;\n}')
    #fout.close()
   
    fout.close()

    #######################
    ## build_prj.tcl
    #######################

    bdtdir = os.path.abspath(os.path.join(filedir, "../bdt_utils"))
    relpath = os.path.relpath(bdtdir, start=cfg['OutputDir'])

    f = open(os.path.join(filedir,'hls-template/build_prj.tcl'),'r')
    fout = open('{}/build_prj.tcl'.format(cfg['OutputDir']),'w')

    for line in f.readlines():

        line = line.replace('nnet_utils', relpath)
        line = line.replace('myproject', cfg['ProjectName'])

        #if 'set_top' in line:
        #    line = line.replace('myproject', '{}_decision_function'.format(cfg['ProjectName']))
        if 'set_part {xc7vx690tffg1927-2}' in line:
            line = 'set_part {{{}}}\n'.format(cfg['XilinxPart'])
        elif 'create_clock -period 5 -name default' in line:
            line = 'create_clock -period {} -name default\n'.format(cfg['ClockPeriod'])
        # Remove some lines
        elif ('weights' in line) or ('-tb firmware/weights' in line):
            line = ''
        elif ('cosim_design' in line):
            line = ''

        fout.write(line)
    f.close()
    fout.close()

def auto_config():
    config = {'ProjectName' : 'my_prj',
              'OutputDir'   : 'my-conifer-prj',
              'Precision'   : 'ap_fixed<18,8>',
              'XilinxPart' : 'xcvu9p-flgb2104-2L-e',
              'ClockPeriod' : '5'}
    return config

def decision_function(X, config):
    np.savetxt('{}/tb_data/tb_input_features.dat'.format(config['OutputDir']),
               X, delimiter=",", fmt='%10f')
    cwd = os.getcwd()
    os.chdir(config['OutputDir'])
    cmd = 'vivado_hls -f build_prj.tcl "csim=1 synth=0" > predict.log'
    success = os.system(cmd)
    if(success > 0):
        print("'predict' failed, check predict.log")
        sys.exit()
    y = np.loadtxt('tb_data/csim_results.log')
    os.chdir(cwd)
    return y

def sim_compile(config):
    return

def build(config):
    cwd = os.getcwd()
    os.chdir(config['OutputDir'])
    cmd = 'vivado_hls -f build_prj.tcl "csim=0 synth=1"'
    success = os.system(cmd)
    if(success > 0):
        print("'build' failed")
        sys.exit()
    os.chdir(cwd)

