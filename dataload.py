from sklearn.model_selection import StratifiedKFold
from graph import get_node_feature
import csv

from torch import nn
import sys
from opt import *
opt = OptInit().initialize()
def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std  = dataset.std()

    return ((dataset - mean) / std).astype(dtype)

def intensityNormalisationFeatureScaling(dataset, dtype):
    max = dataset.max()
    min = dataset.min()

    return ((dataset - min) / (max - min)).astype(dtype)

class dataloader():
    def __init__(self): 
        self.pd_dict = {}
        self.num_classes = opt.num_classes

    def load_data(self):

        subject_IDs = get_ids()
        # Read data, including phenotypic data and labels.
        # It is recommended to adjust the group in the phe file to 0 and 1.
        labels = get_subject_score(subject_IDs, score='Group')
        num_nodes = len(subject_IDs)
        sites= get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        ages = get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        dsms = get_subject_score(subject_IDs, score='DSM_IV_TR') # This indicator contains label information and should not be used.
        fiq = get_subject_score(subject_IDs, score='FIQ')
        genders = get_subject_score(subject_IDs, score='SEX')

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])

        age = np.zeros([num_nodes], dtype=np.float32)
        dsm = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int)
        site = np.zeros([num_nodes], dtype=np.int)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]])-1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            dsm[i] = float(dsms[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]

        # Get the labels and features of the subjects.
        self.y = y 
        self.raw_features = get_node_feature()
        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:,0] = site
        phonetic_data[:,1] = gender
        phonetic_data[:,2] = age
        self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['SEX'] = np.copy(phonetic_data[:,1])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:,2])

        phonetic_score = self.pd_dict
        return self.raw_features, self.y, phonetic_data, phonetic_score
    

    def data_split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=666)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits 

    def get_inputs(self, nonimg, embeddings, phonetic_score):
        # Compute the edges of the HPG.
        S = self.create_type_mask()  # Gender mask matrix
        self.node_ftr = np.array(embeddings.detach().cpu().numpy())
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        aff_adj = get_static_affinity_adj( phonetic_score)  # The Adjacency matrix of the HPG.
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1
        assert flatten_ind == num_edge, "Error in computing edge input"

        # Set the threshold, which is beta in the paper.
        keep_ind = np.where(aff_score > opt.beta)[0]
        edge_index = edge_index[:, keep_ind]
        same_row = []
        same_col = []
        diff_row = []
        diff_col = []
        for i in range(edge_index.shape[1]):
            if S[edge_index[0, i], edge_index[1, i]] == 1:
                same_row.append(edge_index[0, i])
                same_col.append(edge_index[1, i])
            else:
                diff_row.append(edge_index[0, i])
                diff_col.append(edge_index[1, i])

        same_index = np.stack((same_row, same_col)).astype(np.int64)
        diff_index = np.stack((diff_row, diff_col)).astype(np.int64)

        return same_index, diff_index

    def create_type_mask(self):
        subject_list = get_ids()
        num_nodes = len(subject_list)
        type_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        type = get_subject_score(subject_list, score='SEX')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if type[subject_list[i]] == type[subject_list[j]]:
                    type_matrix[i, j] = 1
                    type_matrix[j, i] = 1

        type_matrix = torch.from_numpy(type_matrix)
        device='cuda:0'
        return type_matrix.to(device)

def get_subject_score(subject_list, score):
    scores_dict = {}
    phenotype = opt.phenotype_path
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    return scores_dict


def get_ids(num_subjects=None):
    subject_IDs = np.genfromtxt(opt.subject_IDs_path, dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs

def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]]) 
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]

        if l in ['AGE_AT_SCAN','FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(pd_dict):
    # Phenotypic data similarity scores of HPG based on several phenotypic data were calculated.
    pd_affinity = create_affinity_graph_from_scores(['SEX','SITE_ID'], pd_dict)
    # pd_affinity = create_affinity_graph_from_scores(['AGE_AT_SCAN', 'SEX', 'EDU'], pd_dict)
    pd_affinity = (pd_affinity - pd_affinity.mean(axis=0)) / pd_affinity.std(axis=0)
    return pd_affinity

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, output, target):
        n_classes = output.size(1)
        target_one_hot = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
        target_smooth = target_one_hot * (1 - self.smoothing) + (1 - target_one_hot) * self.smoothing / (n_classes - 1)
        log_probs = nn.functional.log_softmax(output, dim=1)
        loss = nn.functional.kl_div(log_probs, target_smooth, reduction='none').sum(1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass