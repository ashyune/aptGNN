import time
import os
import os.path as osp
import re
import tarfile
import shutil
import glob
from pathlib import Path

def show(str_msg):
    print(str_msg + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

tar_files = [
    'ta1-cadets-e3-official.json.tar.gz',
    'ta1-cadets-e3-official-2.json.tar.gz',
    'ta1-fivedirections-e3-official-2.json.tar.gz',
    'ta1-theia-e3-official-1r.json.tar.gz',
    'ta1-theia-e3-official-6r.json.tar.gz',
    'ta1-trace-e3-official-1.json.tar.gz'
]

for tar_name in tar_files:
    tar_path = osp.join('../graphchi-cpp-master/graph_data/darpatc', tar_name)
    if osp.exists(tar_path):
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall()

path_list = [
    'ta1-cadets-e3-official.json', 
    'ta1-cadets-e3-official-2.json', 
    'ta1-fivedirections-e3-official-2.json', 
    'ta1-theia-e3-official-1r.json', 
    'ta1-theia-e3-official-6r.json', 
    'ta1-trace-e3-official-1.json'
]

pattern_uuid = re.compile(r'uuid\":\"(.*?)\"') 
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')

notice_num = 1000000

for path in path_list:
    id_nodetype_map = {}
    for i in range(100):
        now_path = path + '.' + str(i) if i != 0 else path
        if not osp.exists(now_path): 
            break
        show(now_path)
        cnt = 0
        with open(now_path, 'r') as f:
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    print(cnt)
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: 
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: 
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: 
                    continue
                uuid_match = pattern_uuid.findall(line)
                if not uuid_match:
                    print(line)
                    continue
                uuid = uuid_match[0]
                subject_type = pattern_type.findall(line)

                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        id_nodetype_map[uuid] = 'MemoryObject'
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        id_nodetype_map[uuid] = 'NetFlowObject'
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        id_nodetype_map[uuid] = 'UnnamedPipeObject'
                        continue
                else:
                    id_nodetype_map[uuid] = subject_type[0]
    
    not_in_cnt = 0
    for i in range(100):
        now_path = path + '.' + str(i) if i != 0 else path
        if not osp.exists(now_path): 
            break
        cnt = 0
        with open(now_path, 'r') as f, open(now_path + '.txt', 'w') as fw:
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    print(cnt) 

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    edgeType_match = pattern_type.findall(line)
                    if not edgeType_match: continue
                    edgeType = edgeType_match[0]
                    
                    timestamp_match = pattern_time.findall(line)
                    if not timestamp_match: continue
                    timestamp = timestamp_match[0]
                    
                    srcId = pattern_src.findall(line)
                    if len(srcId) == 0: continue
                    srcId = srcId[0]
                    
                    if srcId not in id_nodetype_map: 
                        not_in_cnt += 1
                        continue
                    srcType = id_nodetype_map[srcId]
                    
                    dstId1 = pattern_dst1.findall(line)
                    if len(dstId1) > 0 and dstId1[0] != 'null':
                        dst1 = dstId1[0]
                        if dst1 not in id_nodetype_map:
                            not_in_cnt += 1
                            continue
                        dstType1 = id_nodetype_map[dst1]
                        fw.write(f"{srcId}\t{srcType}\t{dst1}\t{dstType1}\t{edgeType}\t{timestamp}\n")

                    dstId2 = pattern_dst2.findall(line)
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dst2 = dstId2[0]
                        if dst2 not in id_nodetype_map:
                            not_in_cnt += 1
                            continue
                        dstType2 = id_nodetype_map[dst2]
                        fw.write(f"{srcId}\t{srcType}\t{dst2}\t{dstType2}\t{edgeType}\t{timestamp}\n")

copy_map = {
    'ta1-theia-e3-official-1r.json.txt': '../graphchi-cpp-master/graph_data/darpatc/theia_train.txt',
    'ta1-theia-e3-official-6r.json.8.txt': '../graphchi-cpp-master/graph_data/darpatc/theia_test.txt',
    'ta1-cadets-e3-official.json.1.txt': '../graphchi-cpp-master/graph_data/darpatc/cadets_train.txt',
    'ta1-cadets-e3-official-2.json.txt': '../graphchi-cpp-master/graph_data/darpatc/cadets_test.txt',
    'ta1-fivedirections-e3-official-2.json.txt': '../graphchi-cpp-master/graph_data/darpatc/fivedirections_train.txt',
    'ta1-fivedirections-e3-official-2.json.23.txt': '../graphchi-cpp-master/graph_data/darpatc/fivedirections_test.txt',
    'ta1-trace-e3-official-1.json.txt': '../graphchi-cpp-master/graph_data/darpatc/trace_train.txt',
    'ta1-trace-e3-official-1.json.4.txt': '../graphchi-cpp-master/graph_data/darpatc/trace_test.txt'
}

for src, dst in copy_map.items():
    if osp.exists(src):
        shutil.copy2(src, dst)

for f_path in glob.glob('ta1-*'):
    try:
        Path(f_path).unlink()
    except IsADirectoryError:
        shutil.rmtree(f_path)