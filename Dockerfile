FROM ubuntu:18.04

# Update Ubuntu Software repository
RUN apt update
RUN apt upgrade -y

# Install software 
RUN apt-get install -y git