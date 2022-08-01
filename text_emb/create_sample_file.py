# -*- coding: utf-8 -*-
"""
Created on 2020 5.25

@author: Yudai Pan
"""
import random
import get_type as gt
import numpy as np
import fileinput

def del_first_line(filepath):
    for line in fileinput.input(filepath, inplace=1):
        if not fileinput.isfirstline():
            print(line.replace('\n',''))

def get_train_file(BENCHMARK):
    in_path = "./sampled/" + BENCHMARK + "/"
    # from the sampled data to get the test, train and valid data.
    with open(in_path + 'Fact.txt', 'r') as f:
        total = int(f.readline())
        data_total = [line.strip('\n').split(" ") for line in f.readlines()]
        train_num = int(total * 0.8)
        test_num = int(total * 0.1)
        valid_num = int(total * 0.1)
        sample_list = random.sample(range(0, total), test_num + valid_num + train_num)

        # create test file
        file = open(in_path + 'test2id.txt', 'w')
        file.write(str(test_num) + '\n')
        for index in range(test_num):
            file.write((data_total[sample_list[index]][0]) + ' ' +
                       (data_total[sample_list[index]][1]) + ' ' +
                       (data_total[sample_list[index]][2]) + "\n")
        file.close()

        # create valid file
        file = open(in_path + 'valid2id.txt', 'w')
        file.write(str(valid_num) + '\n')
        for index in range(valid_num):
            file.write((data_total[sample_list[index + test_num]][0]) + ' ' +
                       (data_total[sample_list[index + test_num]][1]) + ' ' +
                       (data_total[sample_list[index + test_num]][2]) + "\n")
        file.close()

        # /////////////////
        file = open(in_path + 'train2id.txt', 'w')
        file.write(str(train_num) + '\n')
        for index in range(train_num):
            file.write((data_total[sample_list[index + test_num + valid_num]][0]) + ' ' +
                       (data_total[sample_list[index + test_num + valid_num]][1]) + ' ' +
                       (data_total[sample_list[index + test_num + valid_num]][2]) + "\n")
        file.close()
        # ////////////////

        f.close()

    # change fact to train2id.txt
    # with open(self.in_path + 'train2id', 'r') as f:
    #     total_fact = int(f.readline())
    #     data_total_fact = [line.strip('\n').split(" ") for line in f.readlines()]
    #     file = open(self.in_path + 'Fact.txt', 'w')
    #     file.write(str(total_fact) + '\n')
    #     for index in range(test_num):
    #         file.write((data_total_fact[index][0]) + ' ' +
    #                    (data_total_fact[index][1]) + ' ' +
    #                    (data_total_fact[index][2]) + "\n")
    #     file.close()

    # get the type file
    gt.get_type(in_path)


def read_entity(filename, ent_dic):
    with open(filename, 'r') as f:
        ents = f.readlines()
        ents = np.delete(ents, 0, 0)
        ents = np.array(ents)
        for ent in ents:
            ent = ent.split()
            ent_dic[ent[1]] = ent[0]


def create_discription_text(BENCHMARK):
    ent_all_dic = {}
    ent_sampled_des_dic = {}
    ent_description_dic = {}
    description_file = "./text_emb/data/WN18/entityWords.txt"
    in_path = "./sampled/" + BENCHMARK + "/"
    entity_filename = "./benchmarks/" + BENCHMARK + '/'

    read_entity(entity_filename + "entity2id.txt", ent_all_dic)

    with open(description_file, 'r') as f:
        ents = f.readlines()
        ents = np.array(ents)
        for ent in ents:
            ent = ent.split("\t")
            ent_description_dic[ent[0]] = [ent[1], ent[2]]


    # save old entity id to dict
    with open(in_path + "entity2id.txt", 'r') as f:
        ents = f.readlines()
        ents = np.delete(ents, 0, 0)
        ents = np.array(ents)
        for ent in ents:
            ent = ent.split()
            if ent[1] in ent_all_dic.keys():
                entity_name = ent_all_dic[ent[1]]
                if entity_name in ent_description_dic.keys():
                    description = ent_description_dic[entity_name]
                else:
                    description = [0,"no \n"]
                ent_sampled_des_dic[entity_name] = description

    with open(in_path + 'train_entity_words.txt', 'w') as f:
        for entity in ent_sampled_des_dic.keys():
            des_num = ent_sampled_des_dic[entity][0]
            des = ent_sampled_des_dic[entity][1]
            f.write(str(entity) + '\t' + str(des_num) + '\t' + str(des))
        f.close()