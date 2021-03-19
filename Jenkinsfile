pipeline {
  environment {
    MGLS_LICENSE_FILE = credentials('msim_licence')
  }
  agent {
    docker {
      image 'conifer-test'
      args  '-v /tools:/tools -e MGLS_LICENSE_FILE=$MGLS_LICENSE_FILE'
    }
  }
  options {
    timeout(time: 1, unit: 'HOURS')
  }
  stages {
    stage('pytest') {
      steps {
        dir(path: 'tests') {
          sh '''#!/bin/bash --login
              source /home/jenkins/miniconda/etc/profile.d/conda.sh
              export PATH=$PATH:/tools/modeltech/bin
              export LIBRARY_PATH=$LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
              source /tools/Xilinx/Vivado/2019.2/settings64.sh
              source /home/jenkins/.bashrc
              conda activate conifer-test
              pip install -U ../ --user
              pytest --cov-report term --cov=conifer --junitxml='reports/report.xml'
              status=$?
              pip uninstall conifer -y
              rm -r prj*
              exit $status'''
        }
      }
    }
  }
  post {
      always {
        junit '**/reports/*.xml' 
      }
  }
}
