#!/bin/bash
echo "CircleCI configuration created at .circleci/config.yml"
echo "Next steps:"
echo "1. Commit this configuration to your repository"
echo "2. Connect your repository to CircleCI"
echo "3. Configure environment variables if needed"

sudo apt-get update
sudo apt-get upgrade -y
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch-$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

#add cloud user to docker group
sudo usermod -aG docker ubuntu
suco usermod -aG docker $USER
newgrp docker

#start configuration of self-hosted machine
#Download the launch agent binary and verify the checksum
mkdir configuration
cd configuration
curl https://raw.githubusercontent.com/circleCI-Public/runner-installation-files/main/download-launch-agent.sh > download-launch-agent.sh
export platform=linux/amd64 && sh ./download-launch-agent.sh


#create the circleci user and working directory
id -u circleci &>/dev/null || sudo adduser --disabled-password --password --gecos GECOS circleci
#but did
id -u circleci &>/dev/null || sudo adduser --disabled-password --gecos "CircleCI user" circleci
sudo mkdir -p /var/opt/circleci
sudo chmod 0750 /var/opt/circleci
sudo chown -R circleci /var/opt/circleci /opt/circleci/circleci


sudo mkdir -p /etc/opt/circleci
sudo touch /etc/opt/circleci/launch-agent-config.yaml
sudo nano /etc/opt/circleci/launch-agent-config.yaml

