#!/bin/bash

# SET PATHS HERE
# Directory where the log files will be written for the jobs
LOGPATH="/sequoia/data2/gvarol/logs/render_`date +%F`"
# A text file with a number in it that determines (dynamically) the maximum parallel jobs allowed
MAXJOBPATH="/sequoia/data1/gvarol/tmp/sequoia.inf"
# Where this code repository is located
CODEPATH="/sequoia/data1/gvarol/surreact/surreact"

if [ -d ${LOGPATH} ]; then
    echo "${LOGPATH} exists."
    read -p "Press enter to continue"
else
    echo "Creating directory at ${LOGPATH}"
    mkdir -p ${LOGPATH}/scripts
    mkdir -p ${LOGPATH}/logs
fi

CODESH=${LOGPATH}/scripts/jobs.txt
for JOBID in {0..1599}; do echo "${JOBID}"; done > $CODESH

for JOBID in {0..1599}
do
    QSUBSTR="#!/bin/bash
#$ -N render_${JOBID}
#$ -l mem_req=1G 
#$ -l h_vmem=12G 
#$ -o ${LOGPATH}/logs/${JOBID}_out.log 
#$ -e ${LOGPATH}/logs/${JOBID}_err.log 
#$ -q bigmem.q,all.q 

cd ${CODEPATH}/datageneration/exe
for r in 0
do
  for v in 0 45 90 135 180 225 270 315
  do
    ./run.sh '--idx ${JOBID} \
      --zrot_euler '\"\$v\"' \
      --repetition '\"\$r\"' \
      --vidlist_path vidlists/uestc/1600seq_surreact_vibe_subset_relative.txt \
      --smpl_result_path ../data/uestc/vibe/train \
      --smpl_estimation_method vibe \
      --bg_path ../data/uestc/backgrounds/ \
      --output_path ../data/surreact/uestc/vibe/ \
      --tmp_path ../data/surreact/uestc/tmp_vibe_output/ \
      --datasetname uestc \
      --split_name train \
      --clothing_option nongrey \
      --with_trans 1 \
      --track_id -1 \
      --cam_height -1 3 \
      --cam_dist 4 6 \
      --fend 300 '
  done
done
"

    QSUBFILE="${LOGPATH}/scripts/submit_${JOBID}.pbs"
    echo "Creating job file ${QSUBFILE}"
    echo "${QSUBSTR}" > ${QSUBFILE}
    # echo "Submitting job ${JOBID}"
    # qsub ${QSUBFILE}
done

# module add sge cluster-tools
QSTAT="qstat"
QSUB="qsub"
USER=gvarol
MAXJOB=$(cat $MAXJOBPATH)

cd $LOGPATH/scripts

while read JOBID
do
    COUNTERJOBS=`$QSTAT | grep gvarol | grep render | sed "s/.* \([0-9][0-9]*\) *$/\1/" | echo \`sum=0; while read line; do let sum=$sum+$line; done; echo $sum\` `
    if [ "$COUNTERJOBS" -ge "$MAXJOB" ]; then
        date "+[%d-%b-%Y %T] Task render : Job number limit reached ($COUNTERJOBS/$MAXJOB slots used)."
       while [ "$COUNTERJOBS" -ge "$MAXJOB" ]; do
           sleep 10
           COUNTERJOBS=`$QSTAT | grep gvarol | grep render | sed "s/.* \([0-9][0-9]*\) *$/\1/" | echo \`sum=0; while read line; do let sum=$sum+$line; done; echo $sum\` `
           NEWMAXJOB=$(cat $MAXJOBPATH)
           if [ "$NEWMAXJOB" -ne "$MAXJOB" ]; then
               MAXJOB=$NEWMAXJOB
               date "+[%d-%b-%Y %T] Task render : Job number limit reached ($COUNTERJOBS/$MAXJOB slots used)."
           fi
        done
    fi
    date "+[%d-%b-%Y %T] Task render : " | tr -d "\n"
    qsub_file="${LOGPATH}/scripts/submit_$JOBID.pbs"
    # echo "Submitting job ${JOBID}"
    $QSUB ${qsub_file}
    sleep 0.1
done < $CODESH

