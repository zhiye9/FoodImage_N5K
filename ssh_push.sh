#!/bin/bash 
eval `ssh-agent -s`
ssh-add -k ~/.ssh/git
