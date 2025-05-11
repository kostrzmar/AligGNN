#! /usr/bin/env python3
import subprocess                                                                                                       
import os                                                                                                               
import time                                                                                                             
import shutil                                                                                                           
import sys     
import errno
import fcntl
import psutil
import platform
import GPUtil
from tabulate import tabulate
from datetime import datetime as dt

if len(sys.argv)<2:
    print("Please provide folder name for experiments")
    sys.exit(errno.EINVAL)
EXPERIMENT_TYPE = sys.argv[1]

ROOT_DIR = "./experiments/"
INPUT_DIR = os.path.join(ROOT_DIR,EXPERIMENT_TYPE)                                        
LOG_DIR = os.path.join(INPUT_DIR , "logs")
PROCESSING_DIR = os.path.join(INPUT_DIR, "processing")                              
DONE_DIR = os.path.join(INPUT_DIR, "done")

TOOLKIT_DIR = "."
TOOLKIT_COMMAND = "align_gnn_toolkit/execute_experiment.py"                                          
COMMAND = ["python", os.path.join(TOOLKIT_DIR, TOOLKIT_COMMAND),"-conf"]                                                
def get_command(file_name, shell=False):                                                                                             
    out = COMMAND.copy()                                                                                                       
    out.append(file_name)                                                                                               
    if shell:
        return " ".join(out)
    return out                                                                                                          

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
        
def update_mem_info(items):
    svmem = psutil.virtual_memory()
    name = f"Memory"
    total = f"{get_size(svmem.total)}"
    used = f"{get_size(svmem.available)}"
    free = f"{get_size(svmem.available)}"
    percent = f"{svmem.percent}%"
    items.append( ("", name, percent, free, used,total, ""))

def update_gpu_info(items):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        # get % percentage of GPU usage of that GPU
        gpu_load = f"{gpu.load*100:.1f}%"
        # get free memory in MB format
        gpu_free_memory = f"{gpu.memoryFree}MB"
        # get used memory
        gpu_used_memory = f"{gpu.memoryUsed}MB"
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        # get GPU temperature in Celsius
        gpu_temperature = f"{gpu.temperature} °C"
        #gpu_uuid = gpu.uuid
        items.append((
            gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
            gpu_total_memory, gpu_temperature#, gpu_uuid
        ))

def update_disk_info(items):
    partitions = psutil.disk_partitions()
    for partition in partitions:
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # this can be catched due to the disk that
            # isn't ready
            continue
        name = f"{partition.device}"
        total = f"{get_size(partition_usage.total)}"
        used = f"{get_size(partition_usage.used)}"
        free = f"{get_size(partition_usage.free)}"
        percent = f"{partition_usage.percent}%"
        items.append((
            "", name, percent, free, used,
            total, ""
        ))

def update_cpu_info(items):
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        percent = f"{psutil.cpu_percent()}%"
        items.append((
            i, f'CPU({i})', percent, "", "",
            "", ""
        ))

def get_cpu_info():
    return "|".join([ f'{x:.1f}%' for x in  psutil.cpu_percent(percpu=True, interval=1)])
    
def get_temp_info():
    items = psutil.sensors_temperatures()
    out = {}
    for item in items.keys():
        for i in items[item]:
            out[f'{item}-{i.label}'] =i.current
    sorted_by_values = dict(sorted(out.items(), key=lambda item: item[1], reverse=True))
    out = ""
    for item in sorted_by_values.keys():
        out+=item + ":"+str(sorted_by_values[item])+"°C|"
    return out

print(f'Scanning folder [{INPUT_DIR}]')
while True:                                                                                                            
    files = list(filter(lambda x:x.endswith('.yaml'), os.listdir(INPUT_DIR)))                                           
    if len(files)>0:                                                                                                    
        file_to_process = files[0]
        start = dt.now()
        print(f'[{dt.now().strftime("%H:%M:%S")}] Start processing of {files[0]} (out of {len(files)})')                                                                   
        shutil.move(os.path.join(INPUT_DIR, file_to_process), os.path.join(PROCESSING_DIR, file_to_process))            
        try:
            cwd = os.getcwd()
            os.chdir(TOOLKIT_DIR)
            results = subprocess.run(get_command(os.path.join(PROCESSING_DIR, file_to_process), shell=True), shell=True, capture_output=True, text=True, check=True)  
            os.chdir(cwd)
        except subprocess.CalledProcessError as e:
            print("Error Code:", e.returncode)
            print("Error msg:", e.stderr)
            print("Error output:", e.stdout)
            raise Exception("Some issues...")   
        if results.returncode ==0:                                                                                      
            shutil.move(os.path.join(PROCESSING_DIR, file_to_process), os.path.join(DONE_DIR, file_to_process))
            stop = dt.now()
            elapsed=stop-start
            print(f'[{dt.now().strftime("%H:%M:%S")}] Done in {elapsed.seconds//60 // 60 % 60:02d}:{elapsed.seconds // 60 % 60:02d}:{elapsed.seconds % 60:02d}') 
        else:                                                                                                           
            raise Exception("Return code:"+str(results.returncode)) 
            
        
        items = []
        update_mem_info(items)
        update_gpu_info(items)
        update_disk_info(items)
        hard_stats = get_temp_info() + "\n" + get_cpu_info() +"\n"
        hard_stats += tabulate(items, headers=("id", "name", "%", "free", "used", "total","temp"))+"\n"
        with open(os.path.join(LOG_DIR, "experiments.txt"), "a") as logs:
            fcntl.flock(logs, fcntl.LOCK_EX)
            logs.write(f'[{start.strftime("%H:%M:%S")} - {stop.strftime("%H:%M:%S")} ({elapsed.seconds//60 // 60 % 60:02d}:{elapsed.seconds // 60 % 60:02d}:{elapsed.seconds % 60:02d})] {file_to_process}\n')
            logs.write(hard_stats)
            logs.write("\n")
            fcntl.flock(logs, fcntl.LOCK_UN)
        print(f'Scanning for new file to process...')
    time.sleep(60)  
    
    

