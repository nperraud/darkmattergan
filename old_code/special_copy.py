import os
from shutil import copyfile
source = '/store/sdsc/sd01/cosmology/ankit/experiment_results/final_checkpoints_summaries'
dest = '/store/sdsc/sd01/cosmology/results_ankit'
for directory in os.listdir(source):
    pathdir = os.path.join(source, directory)
    print('Dealing with {}'.format(pathdir))
    if os.path.isdir(pathdir):
        destdir = os.path.join(dest, directory)
        os.makedirs(destdir, exist_ok=True)
        filelist = []
        timestamps = []
        for file in os.listdir(pathdir):
            pathfile = os.path.join(pathdir, file)
            if os.path.isfile(pathfile):
                filelist.append(pathfile)
                timestamps.append(os.path.getmtime(pathfile))
        to_be_copied = [x for _,x in sorted(zip(timestamps,filelist), reverse=True)][:10]
        for pathfile in to_be_copied:
            copyfile(pathfile, os.path.join(destdir,os.path.basename(pathfile)))
            print('Copy : {}'.format(pathfile))