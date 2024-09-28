import sys
import time
from opt import *
from metrics import accuracy, auc, prf, metrics
from dataload import dataloader
from model import fc_hgnn
import os
from dataload import LabelSmoothingLoss
from dataload import Logger

if __name__ == '__main__':
    # Load preset initial parameters.
    opt = OptInit().initialize()

    # Create log text
    filename = opt.log_path
    log = Logger(filename)
    sys.stdout = log

    # Create a data loader.
    dl = dataloader()
    raw_features, y, nonimg, phonetic_score = dl.load_data()

    # Initializes the data partition for ten-fold cross-validation.
    n_folds = opt.n_folds
    cv_splits = dl.data_split(n_folds)

    # Create training sets, validation sets, and test sets based on 10-fold data.
    train_index, val_index, test_index = [], [], []
    for i in range(len(cv_splits)):
        test_index.append(cv_splits[i][1])
        val_index.append(cv_splits[(1+i)%10][1])
        set = np.setdiff1d(cv_splits[i][0], cv_splits[(1 + i) % 10][1])
        train_index.append(set)

    # Create evaluation metrics
    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    sens = np.zeros(n_folds, dtype=np.float32)
    spes = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)

    # Ten fold-cross verification start !!!
    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))

        # 'fold' value limit written during debugging
        if  fold < 100 :
            print("\r\n========================== Fold {} ==========================".format(fold))
            train_ind = train_index[fold]
            val_ind = val_index[fold]
            test_ind = test_index[fold]
            # The labels and model in this fold.
            labels = torch.tensor(y, dtype=torch.long).to(opt.device)
            model = fc_hgnn(nonimg, phonetic_score).to(opt.device)
            print(model)

            # The loss function in this fold.
            # We found that the use of the label smoothing function provided a modest performance boost.
            # The cross-entropy function is used in this paper.
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn =LabelSmoothingLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
            fold_model_path = opt.ckpt_path + r"\inffus_fold{}.pth".format(fold)


            # The train function.
            def train():
                acc = 0
                for epoch in range(opt.num_iter):
                    # Model training in this epoch.
                    model.train()
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        node_logits= model(raw_features)
                        loss_cla = loss_fn(node_logits[train_ind], labels[train_ind])
                        loss = loss_cla
                        loss.backward()
                        optimizer.step()
                    correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

                    # Model evaluating in this epoch.
                    model.eval()
                    with torch.set_grad_enabled(False):
                        node_logits= model(raw_features)
                    logits_val = node_logits[val_ind].detach().cpu().numpy()
                    correct_val, acc_val = accuracy(logits_val, y[val_ind])
                    val_sen, val_spe = metrics(logits_val, y[val_ind])
                    print("Epoch: {},\tce loss: {:.5f},\tce loss_cla: {:.5f},\ttrain acc: {:.5f},\tval acc: {:.5f},\tval spe: {:.5f},\tval sen: {:.5f}".format(epoch, loss.item(),loss_cla.item(),acc_train.item(),acc_val.item(),val_spe,val_sen),time.localtime(time.time()))
                    if acc_val > acc :
                        acc = acc_val
                        if opt.ckpt_path !='':
                            if not os.path.exists(opt.ckpt_path):
                                os.makedirs(opt.ckpt_path)
                            torch.save(model.state_dict(), fold_model_path)
                            print("{} Saved model to:{}".format("\u2714", fold_model_path))

                    if epoch == (opt.num_iter-1):
                        evaluate()


            # The evaluate function.
            def evaluate():
                print("  Number of testing samples %d" % len(test_ind))
                print('  Start testing...')
                model.load_state_dict(torch.load(fold_model_path))
                model.eval()
                node_logits = model(raw_features)
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
                sens[fold], spes[fold] = metrics(logits_test, y[test_ind])
                aucs[fold] = auc(logits_test,y[test_ind])
                prfs[fold]  = prf(logits_test,y[test_ind])
                print("  Fold {} test accuracy {:.5f}, AUC {:.5f},SEN {:.5f},SPE {:.5f}".format(fold, accs[fold], aucs[fold],sens[fold],spes[fold]))


            # Select the model for training or evaluation.
            if opt.train==1:
                train()
            elif opt.train==0:
                evaluate()
    # Output the final result of ten-fold cross-validation.
    print("\r\n========================== Finish ==========================")
    n_samples = np.array(raw_features).shape[0]
    acc_nfold = np.sum(corrects)/n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs),np.var(accs)))
    print("=> Average test sen in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens),np.var(sens)))
    print("=> Average test spe in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes),np.var(spes)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs),np.var(aucs)))
    pre, sen, f1 = np.mean(prfs,axis=0)
    pre_var, sen_var, f1_var = np.var(prfs,axis=0)
    print("=> Average test pre {:.4f}({:.4f}), recall {:.4f}({:.4f}), F1-score {:.4f}({:.4f})".format(pre,pre_var,sen,sen_var, f1,f1_var))
    print("{} Saved model to:{}".format("\u2714", opt.ckpt_path))




