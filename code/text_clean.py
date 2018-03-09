# Exclude IMF notes for 20090319, 20100627 as the contents of these notes are not relevant to fiscal policy
import requests,re,os
os.chdir("..")
data_dir="data/fiscal_txt"
out_dir="data/text_clean"

doc_list=os.listdir(data_dir)

## Start from Policy Implementation and Effectiveness until 20090924
## Start from Principles for Policy Analysis in 20091107
## Start from (III) Policy Challenges
## Policies from 20111015
## Policy Imperatives from 20121105
## policies for robust global growth in 20130719
## POLICIES FOR JOBS AND GROWTH in 20130905
## POLICIES FOR A SUCCESSFUL RECOVERY in 20140219
## POLICIES TO BOOST GROWTH in 20140917, POLICIES: MANAGING RISKS AND BOOSTING GROWTH, POLICIES: STRONG ACTION IS NEEDED TO RAISE ACTUAL,POLICIES TO RAISE ACTUAL AND POTENTIAL GROWTH,POLICIES


