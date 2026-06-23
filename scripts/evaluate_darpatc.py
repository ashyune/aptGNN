def main():
    eps = 1e-10

    node_map = {}
    with open('id_to_uuid.txt', 'r') as f:
        for line in f:
            parts = line.strip('\n').split(' ')
            if len(parts) >= 2:
                node_map[int(parts[0])] = parts[1]

    gt = {}
    with open('groundtruth_nodeId.txt', 'r') as f_gt:
        for line in f_gt:
            gt[int(line.strip('\n').split(' ')[0])] = 1

    ans = []
    with open('alarm.txt', 'r') as f_alarm:
        for line in f_alarm:
            if line == '\n': 
                continue
            if ':' not in line:
                tot_node = int(line.strip('\n'))
                ans.extend(['tn'] * tot_node)
                for i in gt:
                    if i < len(ans):
                        ans[i] = 'fn'
                continue
            
            line = line.strip('\n')
            parts = line.split(':')
            a = int(parts[0])
            b = parts[1].strip(' ').split(' ')
            flag = 0
            
            for i in b:
                if i == '': 
                    continue
                if int(i) in gt:
                    if int(i) < len(ans):
                        ans[int(i)] = 'tp'
                    flag = 1

            if a in gt:
                if a < len(ans):
                    ans[a] = 'tp'
            else:
                if flag == 0 and a < len(ans):
                    ans[a] = 'fp'

    tn = 0
    tp = 0
    fn = 0
    fp = 0
    
    for i in ans:
        if i == 'tp': tp += 1
        elif i == 'tn': tn += 1
        elif i == 'fp': fp += 1
        elif i == 'fn': fn += 1
        
    print(tp, fp, tn, fn)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fscore = 2 * precision * recall / (precision + recall + eps)
    
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F-Score: ', fscore)

if __name__ == '__main__':
    main()