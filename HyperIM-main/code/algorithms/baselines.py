import math
import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Hyperspreading import Hyperspreading
from tqdm import tqdm
import copy
import random
from transform import Transform
import networkx as nx
import matplotlib
import datetime

matplotlib.use('Agg')
plt.switch_backend('agg')

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # 可以选择"w"
        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal


def getSeeds_sta(degree, i):
    matrix = []
    matrix.append(np.arange(len(degree)))
    matrix.append(degree)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = ['node_index', 'node_degree']
    df_sort_matrix = df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
    degree_list = list(df_sort_matrix.loc['node_degree'])
    nodes_list = list(df_sort_matrix.loc['node_index'])
    chosed_arr = list(df_sort_matrix.loc['node_index'][:i])
    index = np.where(np.array(degree_list) == degree_list[i])[0]
    nodes_set = list(np.array(nodes_list)[index])
    while 1:
        node = random.sample(nodes_set, 1)[0]
        if node not in chosed_arr:
            chosed_arr.append(node)
            break
        else:
            nodes_set.remove(node)
            continue
    return chosed_arr


def degreemax(df_hyper_matrix, K, R, beta, T):
    """
    Degree algorithm
    """
    start_time = time.time()
    sys.stdout = Logger('efficiency//time_'+fileName + '_Degree.txt')
    degree = getTotalAdj(df_hyper_matrix, N)
    for i in range(0, K):
        seeds = getSeeds_sta(degree, i)
        cur_time = time.time()
        run_time = cur_time - start_time
        print(run_time)
    sys.stdout.reset()

    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(0, K):
            seeds = getSeeds_sta(degree, i)
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def HDegree(df_hyper_matrix, K, R, beta, T):
    """
    HDegree algorithm
    """
    start_time = time.time()
    sys.stdout = Logger('efficiency//time_'+fileName + '_HDegree.txt')

    degree = df_hyper_matrix.sum(axis=1)
    ###
    Hdegree = np.array([])
    for item in degree:
        Hdegree = np.append(Hdegree, item)
    ###
    for i in range(0, K):
        seeds = getSeeds_sta(degree, i)
        cur_time = time.time()
        run_time = cur_time - start_time
        print(run_time)
    sys.stdout.reset()

    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(0, K):
            seeds = getSeeds_sta(degree, i)
            # if i == 10:###
            #     starttime = datetime.datetime.now()
            #     for j in range(20):##
            #         scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)##
            #     endtime = datetime.datetime.now()##
            #     print("运行时间为:", endtime - starttime)##
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)

    # print("inf_spread_matrix.shape:", inf_spread_matrix.shape)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    # print("pd.DataFrame(inf_spread_matrix):", pd.DataFrame(inf_spread_matrix))
    return final_scale_list


def getDegreeList(degree):
    matrix = []
    matrix.append(np.arange(len(degree)))
    matrix.append(degree)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = ['node_index', 'node_degree']
    return df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)


def getMaxDegreeNode(degree, seeds):
    degree_copy = copy.deepcopy(degree)
    global chosedNode
    while 1:
        flag = 0
        degree_matrix = getDegreeList(degree_copy)
        node_index = degree_matrix.loc['node_index']
        for node in node_index:
            if node not in seeds:
                chosedNode = node
                flag = 1
                break
        if flag == 1:
            break
    return [chosedNode]


def updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds):
    edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]  # chosenNode所在的Hyperedge集合
    adj_set = []
    for edge in edge_set:
        adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
    adj_set_unique = np.unique(np.array(adj_set))  # adj_set_unique为chosenNode的邻居集合(包含自身)
    for adj in adj_set_unique:
        adj_edge_set = np.where(df_hyper_matrix.loc[adj] == 1)[0]
        adj_adj_set = []
        for each in adj_edge_set:
            adj_adj_set.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
        if adj in adj_adj_set:
            adj_adj_set.remove(adj)
        sum = 0
        for adj_adj in adj_adj_set:
            if adj_adj in seeds:
                sum = sum + 1
        degree[adj] = degree[adj] - sum


def updateDeg_hsd(degree, chosenNode, df_hyper_matrix):
    edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]

    for edge in edge_set:
        node_set = np.where(df_hyper_matrix[edge] == 1)[0]
        for node in node_set:
            degree[node] = degree[node] - 1


def getDegreeWeighted(df_hyper_matrix, N):
    adj_matrix = np.dot(df_hyper_matrix, df_hyper_matrix.T)
    adj_matrix[np.eye(N, dtype=np.bool_)] = 0
    df_adj_matrix = pd.DataFrame(adj_matrix)
    return df_adj_matrix.sum(axis=1)


def getTotalAdj(df_hyper_matrix, N):
    deg_list = []
    nodes_arr = np.arange(N)
    for node in nodes_arr:
        node_list = []
        edge_set = np.where(df_hyper_matrix.loc[node] == 1)[0]
        for edge in edge_set:
            node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
        node_set = np.unique(np.array(node_list))
        deg_list.append(len(list(node_set)) - 1)
    return np.array(deg_list)


def getSeeds_hdd(N, K):
    start_time = time.time()
    seeds = []
    degree = getTotalAdj(df_hyper_matrix, N)
    sys.stdout = Logger('time_' + fileName + '_HSDP.txt')
    for j in range(1, K + 1):
        chosenNode = getMaxDegreeNode(degree, seeds)[0]
        seeds.append(chosenNode)
        cur_time = time.time()
        run_time = cur_time - start_time
        print(run_time)

        updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds)
    sys.stdout.reset()
    return seeds


def getSeeds_hsd(N, K):
    seeds = []
    degree = getTotalAdj(df_hyper_matrix, N)
    for j in range(1, K + 1):
        chosenNode = getMaxDegreeNode(degree, seeds)[0]
        seeds.append(chosenNode)
        updateDeg_hsd(degree, chosenNode, df_hyper_matrix)
    return seeds


def hurDisc(df_hyper_matrix, K, R, N, beta, T):
    """
    HuresticDegreeDiscount algorithm
    """
    start_time = time.time()
    sys.stdout = Logger('efficiency//time_'+fileName + '_HADP.txt')

    inf_spread_matrix = []
    seeds_list = getSeeds_hdd(N, K)
    for i in range(1, K + 1):
        seeds = seeds_list[:i]
        cur_time = time.time()
        run_time = cur_time - start_time
        print(run_time)
    sys.stdout.reset()


    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def sglDisc(df_hyper_matrix, K, R, N, beta, T):
    """
    HuresticSingleDiscount algorithm
    """
    inf_spread_matrix = []
    seeds_list = getSeeds_hsd(N, K)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def generalGreedy(df_hyper_matrix, K, R):
    """
    GeneralGreedy algorithm
    """
    degree = df_hyper_matrix.sum(axis=1)
    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        seeds = []
        for i in range(0, K):
            scale_list_temp = []
            maxNode = 0
            maxScale = 1
            for inode in range(0, len(degree)):
                if inode not in seeds:
                    seeds.append(inode)
                    scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
                    seeds.remove(inode)
                    scale_list_temp.append(scale)
                    if scale > maxScale:
                        maxNode = inode
                        maxScale = scale
            print("maxNode:", maxNode)
            seeds.append(maxNode)
            scale_list.append(max(scale_list_temp))
        # print("seeds:", seeds)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    print("pd.DataFrame(inf_spread_matrix):\n", pd.DataFrame(inf_spread_matrix))
    return final_scale_list


def my_Greedy(df_hyper_matrix, K, R, beta, T):
    start_time = time.time()
    sys.stdout = Logger('efficiency//time_'+fileName + '_Greedy.txt')

    degree = df_hyper_matrix.sum(axis=1)
    inf_spread_matrix = []
    scale_list = []
    seeds = []

    """compute seeds"""
    for i in tqdm(range(0, K)):
        maxNode = 0
        maxScale = 0
        for inode in tqdm(range(0, len(degree))):
            # cur_time = time.time()
            # run_time = cur_time - start_time
            # if run_time >= 86400:  # 24 hours = 86400 seconds
            #     break

            scale_list_temp = []
            if inode not in seeds:
                seeds.append(inode)
                for r in range(500):
                    scale_temp, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
                    scale_list_temp.append(scale_temp)
                seeds.remove(inode)
                iscale = sum(scale_list_temp) / len(scale_list_temp)
                if iscale > maxScale:
                    maxNode = inode
                    maxScale = iscale
        seeds.append(maxNode)
        cur_time = time.time()
        run_time = cur_time - start_time
        print(run_time)
        if run_time >= 86400 * 2:  # 24 hours = 86400 seconds
            break

    sys.stdout.reset()

    """compute influence(scale) of seeds"""
    for i in range(1, len(seeds) + 2):
        tmp_seeds = seeds[0:i]
        scale_list_temp = []
        for j in range(R):
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
        tmp_scale = sum(scale_list_temp) / len(scale_list_temp)
        scale_list.append(tmp_scale)

    return np.array(seeds), np.array(scale_list)


def computeCI(l, N, df_hyper_matrix):
    CI_list = []
    degree = df_hyper_matrix.sum(axis=1)
    M = len(df_hyper_matrix.columns.values)
    for i in range(0, N):
        # 找到它的l阶邻居
        edge_set = np.where(df_hyper_matrix.loc[i] == 1)[0]
        if l == 1:
            node_list = []
            for edge in edge_set:
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
            if i in node_list:
                node_list.remove(i)
            node_set = np.unique(np.array(node_list))
        elif l == 2:
            node_list = []
            for edge in edge_set:
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
            if i in node_list:
                node_list.remove(i)
            node_set1 = np.unique(np.array(node_list))
            node_list2 = []
            edge_matrix = np.dot(df_hyper_matrix.T, df_hyper_matrix)
            edge_matrix[np.eye(M, dtype=np.bool_)] = 0
            df_edge_matrix = pd.DataFrame(edge_matrix)
            adj_edge_list = []
            for edge in edge_set:
                adj_edge_list.extend(list(np.where(df_edge_matrix[edge] != 0)[0]))
            adj_edge_set = np.unique(np.array(adj_edge_list))
            for each in adj_edge_set:
                node_list2.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
            node_set2 = list(np.unique(np.array(node_list2)))
            for node in node_set2:
                if node in list(node_set1):
                    # print(node_set2)
                    node_set2.remove(node)
            node_set = np.array(node_set2)
        ki = degree[i]
        sum = 0
        for u in node_set:
            sum = sum + (degree[u] - 1)
        CI_i = (ki - 1) * sum
        CI_list.append(CI_i)
    return CI_list


def getSeeds_ci(l, N, K, df_hyper_matrix):
    start_time = time.time()
    seeds = []
    n = np.ones(N)
    CI_list = computeCI(l, N, df_hyper_matrix)
    CI_arr = np.array(CI_list)
    for j in range(0, K):
        CI_chosed_val = CI_arr[np.where(n == 1)[0]]
        CI_chosed_index = np.where(n == 1)[0]
        index = np.where(CI_chosed_val == np.max(CI_chosed_val))[0][0]
        node = CI_chosed_index[index]
        n[node] = 0
        seeds.append(node)
    return seeds


def CIAgr(df_hyper_matrix, K, R, N, l, beta, T):
    """
    H-CI algorithm
    """
    inf_spread_matrix = []
    seeds_list = getSeeds_ci(l, N, K, df_hyper_matrix)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def getSeeds_ris(N, K, lamda, theta, df_hyper_matrix):
    start_time1 = time.time()
    S = []
    U = []
    # 迭代θ次
    for theta_iter in range(0, theta):
        df_matrix = copy.deepcopy(df_hyper_matrix)
        # 随机选择节点
        selected_node = random.sample(list(np.arange(len(df_hyper_matrix.index.values))), 1)[0]
        # 以1-λ的比例删边，构成子超图
        all_edges = np.arange(len(df_hyper_matrix.columns.values))
        prob = np.random.random(len(all_edges))
        index = np.where(prob > lamda)[0]
        for edge in index:
            df_matrix[edge] = 0
        # 将子超图映射到普通图
        adj_matrix = np.dot(df_matrix, df_matrix.T)
        adj_matrix[np.eye(N, dtype=np.bool_)] = 0
        df_adj_matrix = pd.DataFrame(adj_matrix)
        df_adj_matrix[df_adj_matrix > 0] = 1
        G = nx.from_numpy_array(df_adj_matrix.values)
        shortest_path = nx.shortest_path(G, target=selected_node)
        RR = []
        for each in shortest_path:
            RR.append(each)
        U.append(list(np.unique(np.array(RR))))
        cur_time1 = time.time()
        run_time1 = cur_time1 - start_time1
        if run_time1 > 172800:
            return S

    # 重复k次
    for k in range(0, K):
        U_list = []
        for each in U:
            U_list.extend(each)
        dict = {}
        for each in U_list:
            if each in dict.keys():
                dict[each] = dict[each] + 1
            else:
                dict[each] = 1
        candidate_list = sorted(dict.items(), key=lambda item: item[1], reverse=True)
        chosed_node = candidate_list[0][0]
        S.append(chosed_node)
        for each in U:
            if chosed_node in each:
                U.remove(each)
        cur_time2 = time.time()
        run_time2 = cur_time2 - start_time1
        if run_time2 > 172800:
            return S

    return S


def RISAgr(df_hyper_matrix, K, R, N, lamda, theta, beta, T):
    """
    H-RIS algorithm
    """
    start_time = time.time()
    sys.stdout = Logger('efficiency//time_'+fileName + '_HRIS.txt')

    inf_spread_matrix = []
    seeds_list = getSeeds_ris(N, K, lamda, theta, df_hyper_matrix)
    cur_time = time.time()
    run_time = cur_time - start_time
    print(run_time)
    sys.stdout.reset()

    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def getNeighbour(targetNode, df_hyper_matrix):
    """
    :param targetNode:
    :param df_hyper_matrix:
    :return: the neighbours of the targetNode
    """
    edge_set = np.where(df_hyper_matrix.loc[targetNode] == 1)[0]  # chosenNode所在的Hyperedge集合
    adj_set = []
    for edge in edge_set:
        adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
    adj_set_unique = np.unique(np.array(adj_set))  # adj_set_unique为chosenNode的邻居集合(包含自身)
    adj_set_unique = list(adj_set_unique)
    if targetNode in adj_set_unique:
        adj_set_unique.remove(targetNode)
    return adj_set_unique  # list


def getPvw(df_hyper_matrix, N, pev, pve):
    Pvw = np.zeros((N, N), dtype=float)
    factor = 1 - pev * pve
    dic_hyperedge = {}
    dic_nbr = {}
    for inode in range(N):
        dic_nbr[inode] = getNeighbour(inode, df_hyper_matrix)
        dic_hyperedge[inode] = np.where(df_hyper_matrix.loc[inode] == 1)[0]  # inode所在的Hyperedge集合

    for inode in range(N):
        inbr = getNeighbour(inode, df_hyper_matrix)
        edge_set1 = dic_hyperedge[inode]  # inode所在的Hyperedge集合
        for jnode in inbr:
            edge_set2 = dic_hyperedge[jnode]  # jnode所在的Hyperedge集合
            inter = np.intersect1d(edge_set1, edge_set2)  # 求交集
            num_common_edges = len(inter)
            Pvw[inode][jnode] = 1 - math.pow(factor, num_common_edges)

    #############################################################
    # """WIC model"""
    # Pvw = np.zeros((N, N), dtype=float)
    # dic_hyperedge = {}
    # dic_nbr = {}
    # for inode in range(N):
    #     dic_nbr[inode] = getNeighbour(inode, df_hyper_matrix)
    #     dic_hyperedge[inode] = np.where(df_hyper_matrix.loc[inode] == 1)[0]     # inode所在的Hyperedge集合
    #
    # for inode in range(N):
    #     P_vw = 0
    #     inbr = getNeighbour(inode, df_hyper_matrix)
    #     edge_set1 = dic_hyperedge[inode]  # inode所在的Hyperedge集合
    #     for jnode in inbr:
    #         edge_set2 = dic_hyperedge[jnode]  # jnode所在的Hyperedge集合
    #         inter = np.intersect1d(edge_set1, edge_set2)  # 求交集
    #         num_common_edges = len(inter)

    return Pvw


def HyperRankLFA(df_hyper_matrix, N, r, Pvw):
    M = np.ones(N)
    for i in range(N - 1, 0, -1):
        for j in range(i):
            M[r[j]] = M[r[j]] + Pvw[r[j]][r[i]] * M[r[i]]
            M[r[i]] = (1 - Pvw[r[j]][r[i]]) * M[r[i]]

    return M


def HyperIMRANK(df_hyper_matrix, N, K, beta, T):
    """
    The Hyper-IMRANK algorithm
    :param df_hyper_matrix:
    :param N:
    :param K:
    :param beta:
    :param T:
    :return:
    """
    start_time = time.time()
    sys.stdout = Logger('efficiency//time_'+fileName + '_HyperIMRANK.txt')

    pev = 0.01
    pve = 0.01
    Pvw = getPvw(df_hyper_matrix, N, pev, pve)
    r_0 = np.arange(N)
    r_1 = r_0
    # M = HyperRankLFA(df_hyper_matrix, N, r_0, Pvw)
    # print(M)
    cnt = 0
    while True:
        cnt += 1
        M = HyperRankLFA(df_hyper_matrix, N, r_0, Pvw)
        r_1 = np.argsort(-M)
        if (r_0 == r_1).all():
            break
        else:
            r_0 = r_1

    seeds = r_1[0:K]
    seeds = seeds.tolist()
    cur_time = time.time()
    run_time = cur_time - start_time
    print(run_time)
    sys.stdout.reset()


    """compute influence(scale) of seeds"""
    scale_list = []
    for i in tqdm(range(1, len(seeds) + 1)):
        tmp_seeds = seeds[0:i]
        scale_list_temp = []
        for j in range(R):
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
        tmp_scale = sum(scale_list_temp) / len(scale_list_temp)
        scale_list.append(tmp_scale)
    return np.array(seeds), np.array(scale_list)


if __name__ == '__main__':
    hs = Hyperspreading()
    tf = Transform()
    file_list = ['Algebra', 'Restaurants-Rev', 'Music-Rev', 'Geometry', 'Bars-Rev', 'NDC-classes-unique-hyperedges', 'iJO1366', 'email-Enron']
    fileName = 'Algebra'
    # fileName = 'Restaurants-Rev'
    # fileName = 'Music-Rev'
    # fileName = 'Geometry'
    # fileName = 'Bars-Rev'
    # fileName = 'NDC-classes-unique-hyperedges'
    # fileName = 'iAF1260b'
    # fileName = 'iJO1366'
    # fileName = 'email-Enron'
    # fileName = 'example'

    df_hyper_matrix, N = tf.changeEdgeToMatrix('../../datasets/' + fileName + '.txt')

    K = 5
    R = 500
    theta = 10000
    beta = 0.02
    T = 10
    # pev = 0.1
    # pve = 0.1

    # alpha = 0.1
    # beta = 0.9
    sta_scale_list = HDegree(df_hyper_matrix, K, R, beta, T)                #Hdegree
    # dmax_scale_list = degreemax(df_hyper_matrix, K, R, beta, T)             #Degree
    # hurd_scale_list = hurDisc(df_hyper_matrix, K, R, N, beta, T)            #HADP (HDD)
    # sgd_scale_list = sglDisc(df_hyper_matrix, K, R, N, beta, T)             #HSDP (HSD)
    # ci_scale_list = CIAgr(df_hyper_matrix, K, R, N, 1, beta, T)             #H-CI(l=1)
    # ci_scale_list2 = CIAgr(df_hyper_matrix, K, R, N, 2, beta, T)            #H-CI(l=2)
    # ris_scale_list = RISAgr(df_hyper_matrix, K, R, N, 0.01, 200, beta, T)     #H-RIS
    #
    final_matrix = []
    # final_matrix.append(hurd_scale_list)                #HADP (HDD)
    # final_matrix.append(sgd_scale_list)                 #HSDP (HSD)
    # final_matrix.append(ris_scale_list)                   #H-RIS
    # final_matrix.append(ci_scale_list)                  #H-CI(l=1)
    # final_matrix.append(ci_scale_list2)                 #H-CI(l=2)
    final_matrix.append(sta_scale_list)                 #Hdegree
    # final_matrix.append(dmax_scale_list)                #Degree

    # final_matrix.append(ggd_scale_list)

    final_df = pd.DataFrame(final_matrix).T
    # # final_df.columns = [['H-Degree', 'RIS', 'Degree', 'CI (l=1)', 'CI (l=2)',
    # #                      'HeuristicSingleDiscount', 'HeuristicDegreeDiscount', 'greedy']]
    # final_df.columns = [['CI (l=2)']]
    #
    # # final_df.columns = [['H-Degree']]
    #
    print(final_df)
    # final_df.to_csv('csv/beta_02_T_10/' + fileName + '_HCI-2.csv')

    seeds, scale_list = my_Greedy(df_hyper_matrix, K, R, beta, T)       #Greedy
    # seeds, scale_list = gxl_algorithm2(df_hyper_matrix, K, 50, N, alpha, beta)
    # seeds, scale_list = my_RIS(df_hyper_matrix, K, R, theta)
    # seeds, scale_list = HyperIMRANK(df_hyper_matrix, N, K, beta, T)
    # for seed in seeds:
    #     print(seed)
    # print("-" * 100)
    # for scale in scale_list:
    #     print(scale)

    # seeds = [101, 52, 15, 12, 121, 76, 29, 7, 156, 55, 37, 0, 136, 32, 27, 18, 158, 50, 35, 51, 196, 17, 95, 170, 204]
    # degree = getTotalAdj(df_hyper_matrix, N)
    # HyperDegree = df_hyper_matrix.sum(axis=1)
    # Hdegree = np.array([])
    # for item in HyperDegree:
    #     Hdegree = np.append(Hdegree, item)
    # for node in seeds:
    #     print(node, ":", Hdegree[node], degree[node])
    # print("Hdegree:")
    # print(np.sort(Hdegree)[393:])
    # print("degree:")
    # print(np.sort(degree)[393:])
    # print(degree[330])
    # print(Hdegree[330])

    # seeds = np.array(seeds)
    # scale_list = np.array(scale_list)

    # np.savetxt(fname="seeds_"+fileName+"_gxl_algorithm2_"+str(alpha)+"_"+str(beta)+".txt", X = seeds, delimiter=',')
    # np.savetxt(fname="scale_list_" + fileName + "_gxl_algorithm2_"+str(alpha)+"_"+str(beta)+".txt", X=scale_list, delimiter=',')
    # np.savetxt(fname="seeds_" + fileName + str(theta) + "_my_RIS.txt", X=seeds, delimiter=',')
    # np.savetxt(fname="scale_list_" + fileName + str(theta) + "_my_RIS.txt", X=scale_list, delimiter=',')

    # np.savetxt(fname="Greedy results/beta_005_T_35/seeds_" + fileName + "_" + "_" + str(beta) + "_" + str(T) + ".txt", X=seeds, delimiter=',')
    # np.savetxt(fname="Greedy results/beta_005_T_35/scale_list_" + fileName + "_" + "_" + str(beta) + "_" + str(T) + ".txt", X=scale_list, delimiter=',')
