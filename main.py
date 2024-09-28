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
        if fold < 100:
            print("\r\n========================== Fold {} ==========================".format(fold))

            # The training set and test set in this fold.
            train_ind = cv_splits[fold][0]
            test_ind = cv_splits[fold][1]

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
                        node_logits = model(raw_features)
                        loss_cla = loss_fn(node_logits[train_ind], labels[train_ind])
                        loss = loss_cla
                        loss.backward()
                        optimizer.step()
                    correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

                    # Model evaluating in this epoch.
                    model.eval()
                    with torch.set_grad_enabled(False):
                        node_logits= model(raw_features)
                    logits_test = node_logits[test_ind].detach().cpu().numpy()
                    correct_test, acc_test = accuracy(logits_test, y[test_ind])
                    test_sen, test_spe = metrics(logits_test, y[test_ind])
                    auc_test = auc(logits_test,y[test_ind])
                    prf_test = prf(logits_test,y[test_ind])

                    # Output the training and test results of this epoch.
                    print("Epoch: {},\tce loss: {:.5f},\tce loss_cla: {:.5f},\ttrain acc: {:.5f},\ttest acc: {:.5f},\ttest spe: {:.5f},\ttest sen: {:.5f}".format(epoch, loss.item(),loss_cla.item(),acc_train.item(),acc_test.item(),test_spe,test_sen),time.localtime(time.time()))

                    # Save the best results and parameters so far.
                    if acc_test > acc:
                        acc = acc_test
                        correct = correct_test
                        aucs[fold] = auc_test
                        prfs[fold] = prf_test
                        sens[fold] = test_sen
                        spes[fold] = test_spe
                        if opt.ckpt_path != '':
                            if not os.path.exists(opt.ckpt_path):
                                os.makedirs(opt.ckpt_path)
                            torch.save(model.state_dict(), fold_model_path)
                            print("{} Saved model to:{}".format("\u2714", fold_model_path))

                # Output the test results of this fold.
                accs[fold] = acc
                corrects[fold] = correct
                print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))


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
                aucs[fold] = auc(logits_test, y[test_ind])
                prfs[fold] = prf(logits_test, y[test_ind])
                print("  Fold {} test accuracy {:.5f}, AUC {:.5f},".format(fold, accs[fold], aucs[fold]))

            # Select the model for training or evaluation.
            if opt.train == 1:
                train()
            elif opt.train == 0:
                evaluate()

    # Output the final result of ten-fold cross-validation.
    print("\r\n========================== Finish ==========================") 
    n_samples = np.array(raw_features).shape[0]
    acc_nfold = np.sum(corrects)/n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs),np.var(accs)))
    print("=> Average test sen in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens),np.var(sens)))
    print("=> Average test spe in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes),np.var(spes)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs),np.var(aucs)))
    se, sp, f1 = np.mean(prfs, axis=0)
    se_var, sp_var, f1_var = np.var(prfs, axis=0)
    print("=> Average test sensitivity {:.4f}({:.4f}), specificity {:.4f}({:.4f}), F1-score {:.4f}({:.4f})".format(se,se_var, sp,sp_var, f1,f1_var))
    print("{} Saved model to:{}".format("\u2714", opt.ckpt_path))




