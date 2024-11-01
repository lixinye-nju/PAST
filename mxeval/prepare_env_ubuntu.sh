#!/usr/bin/bash


printf "%100s" " " | tr ' ' '-'
echo ""
echo "setting up C++ "
printf "%100s" " " | tr ' ' '-'
echo ""
sudo apt update
sudo apt install -y g++


printf "%100s" " " | tr ' ' '-'
echo ""
echo "setting up Java "
printf "%100s" " " | tr ' ' '-'
echo ""
sudo apt-get install openjdk-8-jdk


printf "%100s" " " | tr ' ' '-'
echo ""
echo "setting up Python 3"
printf "%100s" " " | tr ' ' '-'
echo ""
sudo apt update
sudo apt install -y python3 python3-pip
