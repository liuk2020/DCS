for re in 2 3 4 5 6 7
do
    caseName=${re}
    jobName=Surf_${caseName}
    logName=./logs/${jobName}.log
    srun -p hfacnormal02 -N1 -n1 -J ${jobName} python -u oneCase.py ${re} > ${logName} &
    sleep 1
done

wait