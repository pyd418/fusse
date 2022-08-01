# -*- coding: utf-8 -*
import sampling_kg as s
# import sampling_my as s
from models import TransE, TransD, TransH, TransR, RESCAL
import train_embedding as te
# import rule_search_and_learn_weights as r1
import rule_search_my as r
# import rule_search_and_learn_weights_my as r
import numpy as np
import tensorflow as tf
import gc
import time
import csv
import link_prediction as lp
import text_emb.train as te_dkrl
# import ptranse.main as ptranse_1
import os
import random
'''
import sys
sys.stdout.write('\r'+str())
sys.stdout.flush()
'''
IsUncertain = False
BENCHMARK = "FB15K237"
R_minSC = 0.01
R_minHC = 0.001
QR_minSC = 0.5
QR_minHC = 0.001
DEGREE = [R_minSC, R_minHC, QR_minSC, QR_minHC]
Max_rule_length = 2  # not include head atom
_syn = 500
_coocc = 1000
# embedding model parameters
model = TransE.TransE
train_times = 10  # 1000
dimension = 100  # 50
alpha = 0.001  # learning rate
lmbda = 0.01  # degree of the regularization on the parameters
bern = 1  # set negative sampling algorithms, unif(0) or bern(1)
work_threads = 5
nbatches = 150
margin = 1  # the margin for the loss function
u = 0.8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def save_rules(Pt, rule_length, new_index_Pt, candidate, pre_sample):
    print(str(Pt)+":")
    # str(model)[15:21]
    R_num = 0
    QR_num = 0
    with open('./linkprediction/' + BENCHMARK + '/rule_' + str(Pt) + '.txt', 'a+') as f:
        f.write(str(new_index_Pt[1]) + "\n")
        f.write("length: %d, num: %d\n" % (rule_length, len(candidate)))
        i = 1
        rule_ade_list = []
        HC_value_list = []
        for rule in candidate:
            index = rule[0]
            flag = rule[1]
            degree = str(rule[2])
            # Duplicate elimination.
            if rule[2][1] not in HC_value_list:
                rule_ade_list.append(rule)
                HC_value_list.append(rule[2][1])
            # Save Quality rules and rules.
            if flag == 1:
                R_num = R_num + 1
                title = "Rule " + str(i) + ": "
            else:
                QR_num = QR_num + 1
                title = "Qualify Rule " + str(i) + ": "
            line = title + " " + str(index) + " :[SC, HC] " + degree + " "
            for j in range(rule_length):
                # line = line + str(index[j]) + " " + pre_sample[index[j]][1] + "; "
                line = line + pre_sample[index[j]][1] + "; "
            line = line + "\n"
            # print(line)
            f.write(line)
            i = i + 1
        print("\nRule_num: %d" % R_num)
        print("Qualify_Rule_num: %d" % QR_num)
        f.write("\nRule_num: %d\n" % R_num)
        f.write("Qualify_Rule_num: %d\n\n" % QR_num)
    R_num_return = R_num
    QR_num_return = QR_num

    # eliminate duplicate rules
    R_num = 0
    QR_num = 0
    with open('./linkprediction/' + BENCHMARK + '/rule_ade_' + str(Pt) + '.txt', 'a+') as fp:
        fp.write(str(new_index_Pt[1]) + "\n")
        fp.write("length: %d, num: %d\n" % (rule_length, len(rule_ade_list)))
        i = 0
        for rule in rule_ade_list:
            index = rule[0]
            flag = rule[1]
            degree = str(rule[2])
            # Save Quality rules and rules.
            if flag == 1:
                R_num = R_num + 1
                title = "Rule " + str(i) + ": "
            else:
                QR_num = QR_num + 1
                title = "Qualify Rule " + str(i) + ": "
            line = title + " " + str(index) + " :[SC, HC] " + degree + " "
            for j in range(rule_length):
                # line = line + str(index[j]) + " " + pre_sample[index[j]][1] + "; "
                line = line + " " + pre_sample[index[j]][1] + "; "
            line = line + "\n"
            # print(line)
            fp.write(line)
            i = i + 1
        print("\nAfter duplicate elimination, Rule_num: %d" % R_num)
        print("After duplicate elimination, Qualify_Rule_num: %d" % QR_num)
        fp.write("\nAfter duplicate elimination, Rule_num: %d\n" % R_num)
        fp.write("After duplicate elimination, Qualify_Rule_num: %d\n\n" % QR_num)
    return R_num_return, QR_num_return


def predicate_dict(facts):
    time_start = time.time()
    del_flag = 0
    E_i = set()  # Maybe it contains some repeated entities.
    F_i_new = []
    P_count = {}  # After filtering:   Key: p's index; Value: [count]
    # Statistical occurrences.
    P_dic = {}  # Key: p's index; Value: [count, [fact's index list]]
    for i in range(len(facts)):
        f = list(facts[i])
        if f[0] in E_i_1_new or f[1] in E_i_1_new:
            if f[2] in P_dic.keys():
                value = P_dic.get(f[2])
                # Restrict the number of entity!
                # if len(value[1]) <= 50:   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #    value[1].append(i)  # else, only count the freq.
                value[1].append(i)
                P_dic[f[2]] = [value[0] + 1, value[1]]
            else:
                P_dic[f[2]] = [1, [i]]
    # pick out all the predicates
    keys = list(P_dic.keys())
    for key in keys:
        value = P_dic[key]
        # Pick out predicates with high frequency of occurrence.
        if value[0] > 500000:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            del_flag = del_flag + 1
        else:
            P_count[key] = value[0]
            # Get E_i and P_count.
            for j in value[1]:
                E_i.add(facts[j][0])
                E_i.add(facts[j][1])
                if facts[j][3] == 0:
                    F_i_new.append(list(facts[j]))
                    facts[j][3] = 1
    return P_count

def cosSim(x,y):
    '''
    余弦相似度
    '''
    tmp=np.sum(x*y)
    non=np.linalg.norm(x)*np.linalg.norm(y)
    return np.round(tmp/float(non),9)

def manhattanDisSim(x,y):
    '''
    曼哈顿相似度
    '''
    return sum(abs(a-b) for a,b in zip(x,y))

def softmax(x):
    row_max = np.max(x)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s


if __name__ == '__main__':
    begin = time.time()
    print("\nLink Prediction.\nThe benchmark is " + BENCHMARK + ".")
    predicate_all = s.get_pre(filename='./benchmarks/' + BENCHMARK + '/')
    predicate_name = [p[0] for p in predicate_all]

    total_time = 0
    predicate = 0

    # test_Pre_list = np.random.randint(0, predicateSize-1, size=5)
    # test_Pre_list = []
    # if BENCHMARK == "FB15K237":
    #     with open("./benchmarks/" + BENCHMARK + '/237/train/target_pre.txt', 'r') as f:
    #         test_pre_num = f.readline()
    #         test_Pre_list = [int(line.strip('\n')) for line in f.readlines()]

    test_Pre_list = [predicate]     # FB15k

    predict_fact_num_total = 0
    predict_Qfact_num_total = 0
    MRR_total = []
    Hit_10_total = []
    with open(BENCHMARK + ".csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Pt", "len=2", "len=3", "len=4", "Total R num", "Total QR num", "time=2", "time=3", "time=4", "Total time"])
    for Pt in test_Pre_list:
        # Get different train facts for every pt!
        # Here we need to change the directory and look at s.read_data
        if BENCHMARK == "FB15K237":
            facts_old, ent_size_all = s.read_data(filename="./benchmarks/" + BENCHMARK + '/', file_type="train", pt=237)

        if BENCHMARK == "FB15K":
            facts_old, ent_size_all = s.read_data(filename="./benchmarks/" + BENCHMARK + '/', file_type="train", pt=1345)

        if BENCHMARK == "WN18RR":
            facts_old, ent_size_all = s.read_data(filename="./benchmarks/" + BENCHMARK + '/', file_type="train", pt=11)

        # facts_all: has a flag to identify its usage.
        print("Total predicates:%d" % len(predicate_all))

        Pt_start = time.time()
        Pt_i_1 = Pt_start

        # Initialization all the variables.
        num_rule = 0
        num_Qrule = 0
        ent_emb = None
        rel_emb = None
        new_index_Pt = None
        fact_dic_sample = None
        fact_dic_all = None
        facts_sample = None
        ent_size_sample = None
        pre_sample = None
        P_i_list_new = None
        P_count_new = None
        candidate_of_Pt = []
        pre_sample_of_Pt = []
        num_li = []
        time_li = []

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()

        print(Pt)
        print("\n##Begin to sample##\n")
        # After sampling by Pt, return the E_0 and F_0.
        E_0, P_0, F_0, facts_all, E_0_all = s.first_sample_by_Pt(Pt, facts_old)
        # Initialization the sample variables.
        E = E_0
        P = P_0
        F = F_0
        E_i_1_new = E_0
        P_i_list = [list(P_0)]
        max_i = int((Max_rule_length + 1) / 2)
        for length in range(2, Max_rule_length + 1):  # body length
            cur_max_i = int((length + 1) / 2)
            if len(P_i_list) < cur_max_i + 1:

                # If the next P_i hasn't be computed:
                print("\nNeed to compute the P_%d." % cur_max_i)
                E_i, P_i, F_i_new, facts_all, P_count_old = s.sample_by_i(cur_max_i, E_i_1_new, facts_all)

                # Get the next cycle's variable.
                E_i_1_new = E_i - E  # remove the duplicate entity.
                print("The new entity size :%d   (RLvLR need to less than 800.)" % len(E_i_1_new))

                # Merge the result.
                E = E | E_i  # set
                P = P | P_i  # set
                F.extend(F_i_new)
                P_i.add(Pt)
                P_i_list.append(list(P_i))

                # P_count_old dictionary's keys are old indices; P_count dictionary's keys are new indices.
                save_path = './sampled/' + BENCHMARK

                new_index_Pt, P_i_list_new, P_count_new, facts_sample = s.save_and_reindex(length, save_path,
                                                                                           E, P, F, Pt, predicate_name,
                                                                                           P_i_list, P_count_old)
                # P_count = predicate_dict(facts_old)
                # E_old = [i for i in range(ent_size_all)]
                # P_old = [i for i in range(237)]
                # P_i_list_old = [Pt].append(P_old)
                # new_index_Pt, P_i_list_new, P_count_new, facts_sample = s.save_and_reindex(length, save_path,
                #                                                                            set(E_old), set(P_old), facts_old, Pt, predicate_name,
                #                                                                            P_i_list, P_count)

                # The predicates in "pre_sample" is the total number written in file.
                pre_sample = s.get_pre(filename='./sampled/'+BENCHMARK+'/')
                pre_sample_of_Pt.append(pre_sample)
                ent_size_sample = len(E)
                print("\n##End to sample##\n")
                print("\nGet SAMPLE PREDICATE dictionary. (First evaluate on small sample KG.)")
                t = time.time()

                fact_dic_sample = r.RSALW.get_fact_dic_sample(facts_sample)
                fact_dic_all = r.RSALW.get_fact_dic_all(pre_sample, facts_all)
                # fact_dic_sample = facts_sample
                # fact_dic_all = facts_all

                print("fact_dic's key num: %d = %d." % (len(fact_dic_sample), len(fact_dic_all)))
                print("Time: %s \n" % str(time.time() - t))

                # Train Embedding
                print("\n##Begin to train embedding##\n")

                # The parameter of model should be adjust to the best parameters!
                # 0:matrix 1:vector
                ent_emb_1, rel_emb_1 = te.trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha,
                                                 lmbda, bern, margin, model)
                ent_emb_2, rel_emb_2 = te_dkrl.train()
                # ent_emb, rel_emb = ptranse_1.main()

                ent_emb_sum = ent_emb_1 + ent_emb_2
                rel_emb_sum = rel_emb_1 + rel_emb_2
                sim_vec_ent = []
                sim_vec_rel = []

                for i in range(len(ent_emb_1)):
                    s1 = manhattanDisSim(ent_emb_sum[i], ent_emb_1[i])
                    s2 = manhattanDisSim(ent_emb_sum[i], ent_emb_2[i])
                    sim_vec_ent.append(list(softmax(np.array([s1,s2]))))

                for i in range(len(rel_emb_1)):
                    r1 = manhattanDisSim(rel_emb_sum[i], rel_emb_1[i])
                    r2 = manhattanDisSim(rel_emb_sum[i], rel_emb_2[i])
                    sim_vec_rel.append(list(softmax(np.array([r1,r2]))))

                u1_ent = [i[0] for i in sim_vec_ent]
                u2_ent = [i[1] for i in sim_vec_ent]
                u1_rel = [i[0] for i in sim_vec_rel]
                u2_rel = [i[1] for i in sim_vec_rel]

                ent_emb = np.dot(np.diag(u1_ent), ent_emb_1) + np.dot(np.diag(u2_ent), ent_emb_2)
                rel_emb = np.dot(np.diag(u1_rel), rel_emb_1) + np.dot(np.diag(u2_rel), rel_emb_2)

                # rel_emb = (1 - u) * rel_emb_1 + u * rel_emb_2
                # ent_emb = (1 - u) * ent_emb_1 + u * ent_emb_2

                print("\n##End to train embedding##\n")
                isfullKG = False

                # Garbage collection.
                if not gc.isenabled():
                    gc.enable()
                gc.collect()
                gc.disable()
            else:
                print("\nNeedn't to compute the next P_i")
                print("Filter out predicates that appear too frequently to reduce the computational time complexity.\n")
                P_i_list_new, fact_dic_sample, fact_dic_all = s.filter_predicates_by_count(new_index_Pt[0],
                                                                                           P_count_new, P_i_list_new,
                                                                                           fact_dic_sample,fact_dic_all)
                print("After filter, the length of pre: %d :%d " % (len(P_i_list_new[-1]), len(P_count_new)))
                print("##End to sample##")
                print("\n##Begin to train embedding##")
                print("Needn't to train embedding")
                print("##End to train embedding##\n")
                isfullKG = False
                # Garbage collection.
                if not gc.isenabled():
                    gc.enable()
                del P_count_new
                gc.collect()
                gc.disable()
            print("\n##Begin to search and evaluate##\n")

            # Init original object.
            rsalw = r.RSALW()
            rsalw.__int__()
            candidate = rsalw.search_and_evaluate(IsUncertain, 1, length, dimension, DEGREE, new_index_Pt,
                                                  ent_emb, rel_emb, _syn, _coocc, P_i_list_new, isfullKG,
                                                  fact_dic_sample, fact_dic_all, ent_size_sample, ent_size_all,
                                                  E_0_all)
            # candidate=r1.searchAndEvaluate(1, BENCHMARK, new_index_Pt, ent_emb, rel_emb, dimension, ent_size_all, fact_dic_all, DEGREE)
            candidate_of_Pt.extend(candidate)
            # candidate_of_Pt = candidate
            candidate_len = len(candidate)

            print("\n##End to search and evaluate##\n")

            # Save rules and timing.
            R_num, QR_num = save_rules(Pt, length, new_index_Pt, candidate, pre_sample)
            Pt_i = time.time()
            print("\nLength = %d, Time = %f" % (length, (Pt_i-Pt_i_1)))

            # Save in CSV file.
            # num_ade = R_num + QR_num
            num_rule += candidate_len
            num_Qrule += QR_num
            num_li.append(candidate_len)
            time_li.append(Pt_i - Pt_i_1)
            print("There are %d rules." % candidate_len)

            # Send report process E-mail!
            subject = 'xxxx'
            text = "Pt:" + str(Pt) + '\nLength: ' + str(length) + '\n'
            nu = "The number of rules: " + str(candidate_len) + "\n"
            ti = "The time of this length: " + str(Pt_i - Pt_i_1)[0:7] + "\n"

            Pt_i_1 = Pt_i
            text = BENCHMARK + ": " + text + nu + ti
            # Send email.
            # send_process_report_email.send_email_main_process(subject, text)

            # Garbage collection.
            if not gc.isenabled():
                gc.enable()
            del candidate, rsalw
            # del candidate
            gc.collect()
            gc.disable()

        Pt_end = time.time()
        Pt_time = Pt_end - Pt_start
        total_time += Pt_time
        print("\nThis %d th predicate's total Rule num: %d" % (Pt, num_rule))
        print("\nThis %d th predicate's total Quality Rule num: %d" % (Pt, num_Qrule))
        print("This %d th predicate's total time: %f\n\n\n\n" % (Pt, Pt_time))
        line = [Pt]
        line.extend(num_li)
        line.append(num_rule)
        line.append(num_Qrule)
        line.extend(time_li)
        line.append(Pt_time/3600)
            # ["Pt", "len=2", "len=3", "len=4", "Total num", "time=2", "time=3", "time=4", "Total time"]

        # Link prediction.
        # candidate_of_Pt
        # pre_sample_of_Pt
        time_lp_start = time.time()
        print("Begin to predict %d." % Pt)
        lp_save_path = './linkprediction/' + BENCHMARK + '/'
        predict_matrix, predict_fact_num, predict_Qfact_num = lp.predict(lp_save_path, Pt, pre_sample_of_Pt,
                                                                         candidate_of_Pt, facts_all, ent_size_all)
        predict_fact_num_total += predict_fact_num
        predict_Qfact_num_total += predict_Qfact_num
        mid = time.time()
        print("Predict time: %f" % (mid - time_lp_start))
        print('\n')
        print("Begin to test %d." % Pt)
        test_file_path = './benchmarks/' + BENCHMARK + '/'
        MRR, Hit_10 = lp.test(BENCHMARK, test_file_path, lp_save_path, Pt, predict_matrix)
        MRR_total.append([Pt, MRR])
        Hit_10_total.append([Pt, Hit_10])
        mid2 = time.time()
        print("Test time: %f" % (mid2 - mid))
        lp_time = time.time() - time_lp_start
        print("\nLink prediction time: %f" % lp_time)
        hour = int(lp_time / 3600)
        minute = int((lp_time - hour * 3600) / 60)
        second = lp_time - hour * 3600 - minute * 60
        print(str(hour) + " : " + str(minute) + " : " + str(second))

    # Save predict_fact_num, predict_Qfact_num.
    predict_fact_num_total /= len(test_Pre_list)
    predict_Qfact_num_total /= len(test_Pre_list)
    with open('./linkprediction/' + BENCHMARK + '/' + 'predict' + '.txt', 'a') as f:
        f.write("predict_fact_avg: "+str(predict_fact_num_total) + '\n')
        f.write("predict_Qfact_avg: " + str(predict_Qfact_num_total) + '\n')

    # Save MRR, Hit_10 in file.
    with open('./linkprediction/' + BENCHMARK + '/' + 'test' + '.txt', 'a') as f:
        f.write("MRR_total: %d\n" % len(MRR_total))
        for mrr in MRR_total:
            f.write("%d, %f\n" % (mrr[0], mrr[1]))
        f.write("AVG: %f\n" % np.mean([mrr[1] for mrr in MRR_total]))
        f.write("\n")
        f.write("Hit_10_total: %d\n" % len(Hit_10_total))
        for hit10 in Hit_10_total:
            f.write("%d, %f\n" % (hit10[0], hit10[1]))
        f.write("AVG: %f\n" % np.mean([hit10[1] for hit10 in Hit_10_total]))

        # Total time:
        end = time.time() - begin
        hour = int(end / 3600)
        minute = int((end - hour * 3600) / 60)
        second = end - hour * 3600 - minute * 60
        print("\nAlgorithm total time: %f" % end)
        print(str(hour) + " : " + str(minute) + " : " + str(second))
        f.write("Algorithm total time: %d : %d : %f\n" % (hour, minute, second))

