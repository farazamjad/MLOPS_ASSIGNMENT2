pipeline {
  agent any
  options {
    buildDiscarder(logRotator(numToKeepStr: '5'))
  }
  environment {
    DOCKERHUB_CREDENTIALS = credentials('faraz-dockerhub')
  }
  stages {
    stage('Sync Code') {
            steps {
                git branch: 'master', credentialsId: 'farazamjad', url: 'https://github.com/farazamjad/MLOPS_ASSIGNMENT2.git'
            }
        }
    stage('Build') {
      steps {
        sh 'docker build -t model-image:latest .'
      }
    }
    stage('Login') {
      steps {
        sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
      }
    }
    stage('Push Docker Image') {
      steps {
        withDockerRegistry([credentialsId: "faraz-dockerhub", url: "https://index.docker.io/v1/"]) {
          sh  "docker push farazzz/mlops2:latest"
                }
            }
        }
        stage('Run Container') {
      steps {
        sh 'docker run -d -p 5080:80 farazzz/mlops2:latest'
      }
    }
  }


  post {
    always {
      sh 'docker logout'
    }
  }
}
