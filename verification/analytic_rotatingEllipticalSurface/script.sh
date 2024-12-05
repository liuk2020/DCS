for nfp in 2 3 4 5 6
do
    for a in 0.05 0.07
    do
        for delta in 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2
        do
            caseName=${nfp}_${a}_${delta}
            jobName=Surf_${caseName}
            logName=./logs/${jobName}.log
            srun -p hfacnormal02 -N1 -n1 -J ${jobName} python -u rotatingEllipticalSurface.py ${nfp} ${a} ${delta} > ${logName} &
            sleep 1
        done
    done
done

wait
