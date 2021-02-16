pipeline {
  options {
    timeout(time: 1, unit: 'HOURS')
  }
  stages {
    stage('pytest') {
      steps {
        dir(path: 'tests') {
          sh '''#!/bin/bash --login
              source /tools/Xilinx/Vivado/2020.1/settings64.sh
              conda activate conifer-test
              pip install -U ../ --user
              pytest --cov-report term --cov=conifer
              pip uninstall conifer -y'''
        }
      }
    }
  }
  post {
    always {
      dir(path: 'tests') {
          sh '''#!/bin/bash
             ./cleanup.sh'''
      }
    }
  }
}

