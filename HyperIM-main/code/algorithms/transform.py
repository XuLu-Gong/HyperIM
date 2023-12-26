import pandas as pd
import numpy as np
from datamanage import DataManage


class Transform:

    # def changeEdgeToMatrix(self, path):
    #     dm = DataManage()
    #     node_dict, N, M = dm.generateMap(path)
    #     print(node_dict)
    #     matrix = np.random.randint(0, 1, size=(N, M))
    #
    #     df = pd.read_csv(path, index_col=False, header=None)
    #     arr = df.values
    #     index = 0
    #     for each in arr:
    #
    #         edge_list = list(map(int, each[0].split(" ")))
    #         print("edge_list:", edge_list)
    #         for edge in edge_list:
    #
    #             matrix[node_dict[edge]][index] = 1
    #         index = index + 1
    #     print(pd.DataFrame(matrix))
    #     return pd.DataFrame(matrix), N

    def changeEdgeToMatrix(self, path):
        dm = DataManage()
        node_dict, N, M = dm.generateMap(path)
        print(node_dict)
        matrix = np.random.randint(0, 1, size=(N, M))

        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values
        index = 0
        edge_index = np.array([], dtype=int)####
        idx = np.array([], dtype=int)####
        for each in arr:
            edge_list = list(map(int, each[0].split(" ")))
            # edge_list = list(map(int, each[0].split(",")))
            # print("edge_list:", edge_list)
            edge_index = np.append(edge_index, edge_list)###

            for edge in edge_list:
                matrix[node_dict[edge]][index] = 1
                idx = np.append(idx, index)

            index = index + 1

        edge_index = np.vstack((edge_index, idx))

        # print(pd.DataFrame(matrix))
        return pd.DataFrame(matrix), N


